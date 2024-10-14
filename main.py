import argparse
import asyncio
import logging
from collections.abc import Coroutine, Callable
from datetime import datetime
from functools import wraps
from typing import TypeVar, ParamSpec

import aiohttp
import pandas as pd
import pytz
from aioconsole import ainput, aprint
from pydantic import BaseModel, Field, AliasPath
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# --------------Constants------------------------------------

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
PARAMS = {
    "latitude": 55.698520,
    "longitude": 37.359490,
    "current": ["temperature_2m",
                "rain", "showers", "snowfall",
                "pressure_msl",
                "wind_speed_10m", "wind_direction_10m"],
    "timeformat": "unixtime",
    "timezone": "Europe/Moscow",
    "wind_speed_unit": "ms",
    "forecast_minutely_15": 1
}

MOSCOW_TIMEZONE = pytz.timezone('Europe/Moscow')

HPA_TO_MM = 0.7500637554192

DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

DB_URL = 'sqlite+aiosqlite:///weather.db'

logger = logging.getLogger('weather')

T = TypeVar('T')
P = ParamSpec('P')

# ------------Database-------------------------------

engine = create_async_engine(DB_URL)
AsyncSessionmaker = async_sessionmaker(engine)

class Base(DeclarativeBase):
    def as_dict(self) -> dict[str, str | float]:
        return {c.name: getattr(self, c.name) for c in self.__table__.c}

class Weather(Base):
    __tablename__ = 'weather'
    time: Mapped[datetime] = mapped_column(primary_key=True)
    lat: Mapped[float]
    lon: Mapped[float]
    temp: Mapped[float]
    temp_unit: Mapped[str]
    rain: Mapped[float]
    rain_unit: Mapped[str]
    showers: Mapped[float]
    showers_unit: Mapped[str]
    snowfall: Mapped[float]
    snowfall_unit: Mapped[str]
    pressure: Mapped[float]
    pressure_unit: Mapped[str]
    wind_speed: Mapped[float]
    wind_speed_unit: Mapped[str]
    wind_direction: Mapped[float]
    wind_direction_unit: Mapped[str]
    wind_direction_compas: Mapped[str]

# ------------Schema--------------------------------
class WeatherSchema(BaseModel):
    time: datetime = Field(validation_alias=AliasPath('current', 'time'))
    lat: float = Field(alias='latitude')
    lon: float = Field(alias='longitude')
    temp: float = Field(validation_alias=AliasPath('current', 'temperature_2m'))
    temp_unit: str = Field(validation_alias=AliasPath('current_units', 'temperature_2m'))
    rain: float = Field(validation_alias=AliasPath('current', 'rain'))
    rain_unit: str = Field(validation_alias=AliasPath('current_units', 'rain'))
    showers: float = Field(validation_alias=AliasPath('current', 'showers'))
    showers_unit: str = Field(validation_alias=AliasPath('current_units', 'showers'))
    snowfall: float = Field(validation_alias=AliasPath('current', 'snowfall'))
    snowfall_unit: str = Field(validation_alias=AliasPath('current_units', 'snowfall'))
    pressure: float = Field(validation_alias=AliasPath('current', 'pressure_msl'))
    pressure_unit: str = Field(validation_alias=AliasPath('current_units', 'pressure_msl'))
    wind_speed: float = Field(validation_alias=AliasPath('current', 'wind_speed_10m'))
    wind_speed_unit: str = Field(validation_alias=AliasPath('current_units', 'wind_speed_10m'))
    wind_direction: float = Field(validation_alias=AliasPath('current', 'wind_direction_10m'))
    wind_direction_unit: str = Field(validation_alias=AliasPath('current_units', 'wind_direction_10m'))
    wind_direction_compas: str = Field(validation_alias=AliasPath('current', 'wind_direction_10m_compas'))


# ------------Functions------------------------------
def deg_to_compas(deg: int) -> str:
    """Переводит градусы в направление"""
    return DIRECTIONS[int((deg / 45) + 0.5) % 8]

def retry(attempts: int):
    def repeater(f: Callable[P, Coroutine[None, None, T]]) -> Callable[P, Coroutine[None, None, T]]:
        """Повторяет вызов функции в случае неудачи `attempts` раз."""
        async def repeat(*args: P.args, **kwargs: P.kwargs) -> T:
            cnt = 1
            while True:
                try:
                    result = await f(*args, **kwargs)
                    return result
                except Exception as e:
                    logger.debug('Attempt %d: call function %s failed. Exception: %s', cnt, f.__name__, str(e))
                    cnt += 1
                    if cnt > attempts:
                        logger.error('Call function %s failed with exception: %s', f.__name__, str(e))
                        raise asyncio.CancelledError
                    await asyncio.sleep(1)
        return repeat
    return repeater


@retry(5)
async def get_weather() -> WeatherSchema:
    """Функция для получения прогноза погоды с сайта open-meteo
    :return: WeatherSchema - текущий прогноз погоды
    """
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(OPEN_METEO_URL, params=PARAMS, timeout=3) as response:
            res = await response.json()
    # сконвертируем время, давление и направление ветра
    res['current']['time'] = datetime.fromtimestamp(res['current']['time'], tz=MOSCOW_TIMEZONE)
    res['current']['pressure_msl'] = round(res['current']['pressure_msl'] * HPA_TO_MM, 1)
    res['current_units']['pressure_msl'] = 'mmHg'
    res['current']['wind_direction_10m_compas'] = deg_to_compas(res['current']['wind_direction_10m'])
    res_schema = WeatherSchema.model_validate(res)
    return res_schema


@retry(2)
async def save_to_db(schema: WeatherSchema):
    async with AsyncSessionmaker() as session:
        weather = Weather(**schema.model_dump())
        await session.merge(weather)
        await session.commit()

async def weather_forecast(interval: int):
    """Get weather forecast with some interval

    :param interval - interval for requests in minutes
    """
    interval_sec = interval * 60
    try:
        while True:
            weather = await get_weather()
            await save_to_db(weather)
            await asyncio.sleep(interval_sec)
    except asyncio.CancelledError:
        logger.debug('Weather courutine canceled')

@retry(1)
async def read_weather_from_db() -> list[dict[str, str | float]]:
    """Read last 10 records from database and return it as dicts"""
    async with AsyncSessionmaker() as session:
        rows = await session.scalars(select(Weather).order_by(Weather.time.desc()).limit(10))
        dicts = [r.as_dict() for r in rows]
        logger.debug('Read %d records from database', len(dicts))
    return dicts

@retry(1)
async def write_to_excel(fname: str, data: list[dict[str, str | float]]):
    """Write data to file

    :param fname: file name, must end with .xlsx
    :param data: list of records from database
    :return: None
    """
    df = pd.DataFrame.from_records(data)
    loop = asyncio.get_running_loop()
    logger.debug('Writing weather to file %s', fname)
    await loop.run_in_executor(None, df.to_excel, fname)
    logger.info('%d rows was written to the file %s successfully', len(df), fname)

async def get_input_and_save():
    """Запрашивает имя файла и сохраняет последние 10 записей в него"""
    await aprint('You can store data from database.\n'
                 'Enter file name without extention or press <Enter>\n'
                 'for default filename (date and time).\n'
                 'Write <q> to exit the program.\n'
                 )
    try:
        while True:
            name = await ainput('Write file name or <q> and press <Enter>:')
            if name == 'q':
                break
            name = name or datetime.now().strftime('%d%m%Y_%H%M')
            name += '.xlsx'
            data = await read_weather_from_db()
            await write_to_excel(name, data)
    except asyncio.CancelledError:
        logger.debug('Console courutine canceled')

def parse_args() -> argparse.Namespace:
    """Функция для получения аргументов из командной строки"""
    parser = argparse.ArgumentParser(
        prog='Weather forecast',
        description='Script, that sends requests to open-meteo, stored it in db or file.',
    )
    parser.add_argument(
        '-i', '--interval',
        help='Interval in minutes to send requests (default=15)',
        default=15,
        type=int
    )
    return parser.parse_args()

async def main():
    async with engine.connect() as conn:
        await conn.run_sync(Base.metadata.create_all)
    args = parse_args()
    weather_task = asyncio.create_task(weather_forecast(args.interval))
    console_task = asyncio.create_task(get_input_and_save())
    done, pending = await asyncio.wait((weather_task, console_task), return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending)



if __name__ == '__main__':
    logging.basicConfig(filename='logs.txt', filemode='w', level=logging.INFO)
    asyncio.run(main())