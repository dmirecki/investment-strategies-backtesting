from datetime import date
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def run_simulation(df: pd.DataFrame,
                   distribution: List[Tuple[str, str, float]],
                   initial_investment_value: float,
                   rebalancing: Optional[str] = 'yearly') -> pd.DataFrame:
    current = {}
    values_in_time = pd.DataFrame(
        columns=[money_in_good for _, money_in_good, _ in distribution] + ['investment_value'])

    for date, row in tqdm(df.iterrows(), total=len(df)):
        if len(current) == 0:  # start of investments
            for return_rate, money_in_good, pct in distribution:
                current[money_in_good] = initial_investment_value * pct
        elif rebalancing == 'yearly' and date.month == 1:  # yearly rebalancing
            current_investment_value = sum(current.values())
            for return_rate, money_in_good, pct in distribution:
                current[money_in_good] += (current_investment_value * pct - current[money_in_good])

        for return_rate, money_in_good, pct in distribution:
            current[money_in_good] *= (1 + row[return_rate])

        current_investment_value = sum(current.values())
        values_in_time.loc[date] = current
        values_in_time.loc[date, 'investment_value'] = current_investment_value

    return values_in_time['investment_value']


def calculate_maximum_drawdown(investment_values: pd.Series) -> Tuple[float, date, date]:
    drawdowns = (investment_values.cummax() - investment_values[::-1].cummin()) / investment_values.cummax()

    ix_drawdown_start = drawdowns.argmax()
    ix_drawdown_end = investment_values[investment_values.index[ix_drawdown_start]:].argmin()

    return (
        drawdowns.max(),
        investment_values.index[ix_drawdown_start].date(),
        investment_values.index[ix_drawdown_start + ix_drawdown_end].date()
    )


def calculate_yearly_return_rate(investment_values: pd.Series) -> float:
    years = np.ceil((investment_values.index[-1] - investment_values.index[0]).days / 365)
    return_rate = (investment_values.iloc[-1] - investment_values.iloc[0]) / investment_values.iloc[0]

    yearly_return_rate = np.power(1 + return_rate, 1 / years) - 1

    return yearly_return_rate
