"""
SQLAlchemy ORM models for the explainable DBMS schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""


class Customer(Base):
    """Customer demographic and credit profile."""

    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(20), nullable=False)
    location: Mapped[str] = mapped_column(String(255), nullable=False)
    income: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    credit_score: Mapped[int] = mapped_column(Integer, nullable=False)

    transactions: Mapped[list["Transaction"]] = relationship("Transaction", back_populates="customer")
    predictions: Mapped[list["PredictionResult"]] = relationship("PredictionResult", back_populates="customer")


class Product(Base):
    """Product catalog."""

    __tablename__ = "products"

    product_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    cost: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    supplier: Mapped[str] = mapped_column(String(255), nullable=False)

    transactions: Mapped[list["Transaction"]] = relationship("Transaction", back_populates="product")


class Transaction(Base):
    """Customer-product transactions."""

    __tablename__ = "transactions"

    transaction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.product_id"), nullable=False)
    transaction_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    total_amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    discount: Mapped[float] = mapped_column(Float, nullable=False)
    payment_method: Mapped[str] = mapped_column(String(50), nullable=False)

    customer: Mapped[Customer] = relationship("Customer", back_populates="transactions")
    product: Mapped[Product] = relationship("Product", back_populates="transactions")


class PredictionResult(Base):
    """Persisted predictions and associated metadata."""

    __tablename__ = "prediction_results"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instance_id: Mapped[str] = mapped_column(String(255), nullable=False)
    prediction_type: Mapped[str] = mapped_column(String(100), nullable=False)
    prediction_value: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    prediction_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=True)
    shap_values: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lime_values: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

