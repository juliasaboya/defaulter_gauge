"""Aquisição e validação do dataset UCI 350."""
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests
import xlrd
from ucimlrepo import fetch_ucirepo

from .features import BILL, DELAY, LIMIT, PAID

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'credit_card_clients.csv'
URL = 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'
TARGET = 'default_next_month'
ORIGINAL_TARGET = 'default payment next month'
PREDICTORS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    *[f'BILL_AMT{i}' for i in range(1, 7)],
    *[f'PAY_AMT{i}' for i in range(1, 7)],
]


STATUS = ('PAY_3', 'PAY_2', 'PAY_0')
BILLS = ('BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1')
PAYMENTS = ('PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1')
INPUTS = ['LIMIT_BAL', *STATUS, *BILLS, *PAYMENTS]
UCI_HISTORY = dict(zip(INPUTS, (LIMIT, *DELAY, *BILL, *PAID)))


def uci_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Mapeia julho, agosto e setembro para mês 1, 2 e 3 do histórico."""
    return frame.loc[:, INPUTS].rename(columns=UCI_HISTORY)


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={ORIGINAL_TARGET: TARGET}).loc[:, ['ID', *PREDICTORS, TARGET]]
    frame = frame.apply(pd.to_numeric, errors='raise')
    if len(frame) != 30_000 or frame.isna().any().any() or not frame['ID'].is_unique:
        raise ValueError('Esperados 30.000 IDs únicos e dados completos.')
    if frame[TARGET].value_counts().to_dict() != {0: 23_364, 1: 6_636}:
        raise ValueError('Distribuição de classes diferente da esperada na base UCI.')
    return frame


def read_zip(content: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(content)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith('.xls')]
        if len(members) != 1:
            raise ValueError('Esperada uma planilha XLS no ZIP da UCI.')
        xls = archive.read(members[0])
    with xlrd.open_workbook(file_contents=xls) as workbook:
        sheet = workbook.sheet_by_index(0)
        for row in range(min(10, sheet.nrows)):
            header = sheet.row_values(row)
            if {'ID', 'LIMIT_BAL', ORIGINAL_TARGET}.issubset(header):
                records = (sheet.row_values(index) for index in range(row + 1, sheet.nrows))
                return pd.DataFrame(records, columns=header)
    raise ValueError('Cabeçalho do XLS da UCI não encontrado.')


def download() -> None:
    try:
        response = requests.get(URL, timeout=(10, 60))
        response.raise_for_status()
        frame = validate(read_zip(response.content))
        source = 'ZIP/XLS oficial da UCI'
    except (requests.RequestException, BadZipFile, xlrd.XLRDError, ValueError, KeyError) as error:
        print(f'Falha no ZIP/XLS ({error}). Tentando a biblioteca oficial ucimlrepo.')
        try:
            dataset = fetch_ucirepo(id=350)
        except ConnectionError as exc:
            raise SystemExit(f'Não foi possível obter a base oficial da UCI: {exc}') from exc
        # A versão tabular também usa os nomes X1–X23 em algumas distribuições.
        aliases = {f'X{i}': name for i, name in enumerate(PREDICTORS, 1)}
        frame = validate(dataset.data.original.rename(columns={**aliases, 'Y': TARGET}))
        source = 'ucimlrepo (UCI 350)'
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=DATA.parent, suffix='.csv', delete=False) as output:
        temporary = Path(output.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(DATA)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'{source} → {DATA} | Classes: {frame[TARGET].value_counts().to_dict()}')


if __name__ == '__main__':
    download()
