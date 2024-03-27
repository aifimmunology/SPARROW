#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import dask.dataframe as dd
import numpy as np
def sum_per_cell(cellByGeneDf,exclude_blanks=False):
    """
    compute column wise sum of all transcripts on a per cell basis

    Parameters: cellByGeneDf: DataFrame
                dask or pandas m x n dataframe (cell (m) by gene (n) )

                exclude_blanks:bool, default=True
                whether to include only real, non-blank genes when computing per cell sums
    """
    try:
        assert (type(cellByGeneDf)==dd.core.DataFrame or type(cellByGeneDf)==pd.core.frame.DataFrame)
        if type(cellByGeneDf)==dd.core.DataFrame:
            if cellByGeneDf.columns.dtype==int:
                columns=cellByGeneDf.columns.astype('str')
            else:
                columns=cellByGeneDf.columns
            if exclude_blanks:
                return cellByGeneDf.iloc[:,np.where(~columns.str.contains('Blank'))[0]].sum(axis=1).compute()
            else:
                return cellByGeneDf.sum(axis=1).compute()
        else:
            if cellByGeneDf.columns.dtype==int:
                columns=cellByGeneDf.columns.astype('str')
            else:
                columns=cellByGeneDf.columns            
            if exclude_blanks:
                return cellByGeneDf.iloc[:,np.where(~columns.str.contains('Blank'))[0]].sum(axis=1)
            else:
                return cellByGeneDf.sum(axis=1)
    except AssertionError as e:
        print (e, 'input datatype should be one of dask dataframe or pandas dataframe')


def sum_per_trx(cellByGeneDf):
    """
    compute row wise sum of all transcripts

    Parameters: cellByGeneDf: DataFrame
                dask or pandas m x n dataframe (cell (m) by gene (n) )
    """
    try:
        assert (type(cellByGeneDf)==dd.core.DataFrame or type(cellByGeneDf)==pd.core.frame.DataFrame)
        if type(cellByGeneDf)==dd.core.DataFrame:
            return cellByGeneDf.sum(axis=0).compute()
        else:
            return cellByGeneDf.sum(axis=0)
    except AssertionError as e:
        print (e, 'input datatype should be one of dask dataframe or pandas dataframe')
