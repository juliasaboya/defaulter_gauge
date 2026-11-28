"""Transformações ajustadas dentro de cada partição de treinamento."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif

from .features import MONTHS

DELAY_SEVERITY = {f'atraso_{month}' for month in MONTHS}


class SelectIndicators(TransformerMixin, BaseEstimator):
    """Prioriza associação univariada e elimina redundâncias dentro das classes."""

    def __init__(self, columns, max_correlation=0.85):
        self.columns = columns
        self.max_correlation = max_correlation

    def fit(self, X, y):
        data = X.loc[:, self.columns]
        data = data.loc[:, data.nunique() > 1]
        if data.shape[1] == 0:
            raise ValueError('Nenhum indicador candidato varia no conjunto de treinamento.')
        relevance, _ = f_classif(data, y)
        ranking = sorted(zip(data.columns, relevance), key=lambda item: -item[1])
        within_class = [
            data.loc[np.asarray(y) == label].corr(method='spearman').abs().fillna(0)
            for label in np.unique(y)
        ]
        correlation = np.maximum.reduce([matrix.to_numpy() for matrix in within_class])
        positions = {name: index for index, name in enumerate(data.columns)}
        selected = []
        removed = {}
        for name, _ in ranking:
            duplicate = next((
                peer for peer in selected
                if correlation[positions[name], positions[peer]] >= self.max_correlation
            ), None)
            if duplicate is None:
                selected.append(name)
            else:
                removed[name] = {
                    'retained': duplicate,
                    'max_within_class_spearman': float(correlation[positions[name], positions[duplicate]]),
                }
        self.selected_ = selected
        self.relevance_ = {name: float(score) for name, score in ranking}
        self.redundant_ = removed
        return self

    def transform(self, X):
        return X.loc[:, self.selected_]


class IndicatorBins(TransformerMixin, BaseEstimator):
    """Discretiza atributos contínuos e preserva níveis observáveis em campos discretos."""

    def __init__(self, n_bins=8):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        self.edges_ = []
        for name, values in X.items():
            if name in DELAY_SEVERITY:
                self.edges_.append(np.arange(9) + 0.5)
                continue
            levels = np.unique(values)
            if len(levels) <= self.n_bins:
                # Intervalos entre os níveis. A previsão pode receber níveis ausentes no treino.
                cuts = (levels[1:] + levels[:-1]) / 2
            else:
                quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                cuts = np.unique(np.quantile(values, quantiles))
                # Uma massa no menor valor pode ocupar todos os quantis.
                cuts = cuts[cuts > levels[0]]
                if cuts.size == 0:
                    cuts = np.array([(levels[0] + levels[1]) / 2])
            self.edges_.append(cuts)
        return self

    def transform(self, X):
        return np.column_stack([
            np.searchsorted(cuts, values, side='right')
            for cuts, (_, values) in zip(self.edges_, X.items())
        ])
