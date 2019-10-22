from sqlalchemy import Column, String, Integer, Float, Date, ForeignKey, PickleType
from sqlalchemy.orm import relationship, backref

from base import Base


class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True)
    date = Column(Date)
    sipm = relationship("SiPM", uselist=False, back_populates="measurements")
    photodiode = relationship("Photodiode", uselist=False, back_populates="measurements")
    led = relationship("LED", uselist=False, back_populates="measurements")


class SiPM(Base):
    __tablename__ = "sipms"
    id = Column(Integer, primary_key=True)
    bias = Column(PickleType)
    dark_current = Column(PickleType)
    light_current = Column(PickleType)


class Photodiode(Base):
    __tablename__ = "photodiodes"
    id = Column(Integer, primary_key=True)
    dark_current = Column(PickleType)
    light_current = Column(PickleType)


class Led(Base):
    __tablename__ = "leds"
    id = Column(Integer, primary_key=True)
    wavelength = Column(Float)