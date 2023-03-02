# Cryptocurrencies
Use unsupervised learning algorithms (PCA & K-Means) to predict cryptocurrency investments.


## Overview
Use unsupervised learning algorithms (PCA & K-Means) to on what cryptocurrencies are on the trading market and how they could be grouped to create a classification system for this new investment.

Overview of Steps: 
1. Preprocessing the Data for PCA
2. Reducing Data Dimensions Using PCA
3. Clustering Cryptocurrencies Using K-means 
4. Visualizing and Evaluate Cryptocurrencies Results 


## Resources

* Data Source: crypto_data.csv, https://min-api.cryptocompare.com/data/all/coinlist
* ML Libraries: sklearn, plotly, hvplot, matplotlib
* Python 3.11
* Jupyter NB

## Results: 


# Clustering Crypto

```python
# Initial imports
import pandas as pd
import hvplot.pandas
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

```

### Deliverable 1: Preprocessing the Data for PCA


```python
# Load the crypto_data.csv dataset.
file_path = pd.read_csv('crypto_data.csv')
crypto_df = pd.DataFrame(file_path)
crypto_df.sample(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>IsTrading</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1194</th>
      <td>LSK</td>
      <td>Lisk</td>
      <td>DPoS</td>
      <td>True</td>
      <td>DPoS</td>
      <td>1.200121e+08</td>
      <td>159918400</td>
    </tr>
    <tr>
      <th>481</th>
      <td>PSI</td>
      <td>PSIcoin</td>
      <td>X11</td>
      <td>True</td>
      <td>PoS</td>
      <td>NaN</td>
      <td>696969</td>
    </tr>
    <tr>
      <th>436</th>
      <td>MUDRA</td>
      <td>MudraCoin</td>
      <td>X13</td>
      <td>True</td>
      <td>PoS</td>
      <td>5.000000e+06</td>
      <td>200000000</td>
    </tr>
    <tr>
      <th>1224</th>
      <td>ADM</td>
      <td>Adamant</td>
      <td>DPoS</td>
      <td>True</td>
      <td>DPoS</td>
      <td>NaN</td>
      <td>200000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CETI</td>
      <td>CETUS Coin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>598</th>
      <td>CHIEF</td>
      <td>TheChiefCoin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>NaN</td>
      <td>2500000000</td>
    </tr>
    <tr>
      <th>895</th>
      <td>CROAT</td>
      <td>Croat</td>
      <td>CryptoNight</td>
      <td>True</td>
      <td>PoW</td>
      <td>NaN</td>
      <td>100467441</td>
    </tr>
    <tr>
      <th>776</th>
      <td>BMXT</td>
      <td>Bitmxittz</td>
      <td>Scrypt</td>
      <td>False</td>
      <td>PoW/PoS</td>
      <td>NaN</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>823</th>
      <td>INN</td>
      <td>Innova</td>
      <td>NeoScrypt</td>
      <td>True</td>
      <td>PoW</td>
      <td>6.375259e+06</td>
      <td>45000000</td>
    </tr>
    <tr>
      <th>297</th>
      <td>TAM</td>
      <td>TamaGucci</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS/PoC</td>
      <td>NaN</td>
      <td>5300000</td>
    </tr>
  </tbody>
</table>
</div>




```python
crypto_df = crypto_df.set_index('Unnamed: 0')
crypto_df.index.name = ''
```


```python
# Keep all the cryptocurrencies that are being traded.
crypto_df = crypto_df[crypto_df.IsTrading == True]
crypto_df.sample(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>IsTrading</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ECA</th>
      <td>Electra</td>
      <td>NIST5</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>2.839915e+10</td>
      <td>30000000000</td>
    </tr>
    <tr>
      <th>XBOT</th>
      <td>SocialXbotCoin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>0.000000e+00</td>
      <td>2000000</td>
    </tr>
    <tr>
      <th>ACID</th>
      <td>AcidCoin</td>
      <td>SHA-256</td>
      <td>True</td>
      <td>PoW</td>
      <td>NaN</td>
      <td>4500000000</td>
    </tr>
    <tr>
      <th>BENJI</th>
      <td>BenjiRolls</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW</td>
      <td>2.027610e+07</td>
      <td>35520400</td>
    </tr>
    <tr>
      <th>XG</th>
      <td>XG Sports</td>
      <td>XG Hash</td>
      <td>True</td>
      <td>PoW/PoS</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>EXP</th>
      <td>Expanse</td>
      <td>Ethash</td>
      <td>True</td>
      <td>PoW</td>
      <td>1.049528e+07</td>
      <td>16906397</td>
    </tr>
    <tr>
      <th>GBRC</th>
      <td>GBR Coin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW</td>
      <td>0.000000e+00</td>
      <td>87500000</td>
    </tr>
    <tr>
      <th>MADC</th>
      <td>MadCoin</td>
      <td>Scrypt</td>
      <td>True</td>
      <td>PoW</td>
      <td>NaN</td>
      <td>10000000</td>
    </tr>
    <tr>
      <th>WAGE</th>
      <td>Digiwage</td>
      <td>Quark</td>
      <td>True</td>
      <td>PoS</td>
      <td>2.729968e+07</td>
      <td>120000000</td>
    </tr>
    <tr>
      <th>UMO</th>
      <td>Universal Molecule</td>
      <td>Blake</td>
      <td>True</td>
      <td>PoW</td>
      <td>1.578281e+06</td>
      <td>105120001.44</td>
    </tr>
  </tbody>
</table>
</div>




```python
crypto_df.shape
```




    (1144, 6)




```python
# Remove the "IsTrading" column. 
crypto_df = crypto_df.drop(['IsTrading'], axis = 1)
crypto_df.shape
```




    (1144, 5)




```python
# Remove rows that have at least 1 null value.
crypto_df = crypto_df.dropna()
crypto_df.shape
```




    (685, 5)




```python
# Keep the rows where coins are mined.
crypto_df = crypto_df.loc[crypto_df['TotalCoinsMined'] > 0]
crypto_df.shape
```




    (532, 5)




```python
# Create a new DataFrame that holds only the cryptocurrencies names.
crypto_names = crypto_df[['CoinName']]
crypto_names.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>EliteCoin</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>Bitcoin</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethereum</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the crypto_df DataFrame index is the same index for this new DataFrame.
crypto_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CoinName</th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>42 Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
    </tr>
    <tr>
      <th>404</th>
      <td>404Coin</td>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>EliteCoin</td>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>Bitcoin</td>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethereum</td>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.076842e+08</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
crypto_df = crypto_df.drop(['CoinName'], axis = 1)
crypto_df.shape
```




    (532, 4)




```python
crypto_df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
    </tr>
    <tr>
      <th>404</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.076842e+08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LTC</th>
      <td>Scrypt</td>
      <td>PoW</td>
      <td>6.303924e+07</td>
      <td>84000000</td>
    </tr>
    <tr>
      <th>DASH</th>
      <td>X11</td>
      <td>PoW/PoS</td>
      <td>9.031294e+06</td>
      <td>22000000</td>
    </tr>
    <tr>
      <th>XMR</th>
      <td>CryptoNight-V7</td>
      <td>PoW</td>
      <td>1.720114e+07</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ETC</th>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.133597e+08</td>
      <td>210000000</td>
    </tr>
    <tr>
      <th>ZEC</th>
      <td>Equihash</td>
      <td>PoW</td>
      <td>7.383056e+06</td>
      <td>21000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Use get_dummies() to create variables for text features.
X = pd.get_dummies(crypto_df, columns = ['Algorithm', 'ProofType'])
X
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
      <th>Algorithm_1GB AES Pattern Search</th>
      <th>Algorithm_536</th>
      <th>Algorithm_Argon2d</th>
      <th>Algorithm_BLAKE256</th>
      <th>Algorithm_Blake</th>
      <th>Algorithm_Blake2S</th>
      <th>Algorithm_Blake2b</th>
      <th>Algorithm_C11</th>
      <th>...</th>
      <th>ProofType_PoW/PoS</th>
      <th>ProofType_PoW/PoS</th>
      <th>ProofType_PoW/PoW</th>
      <th>ProofType_PoW/nPoS</th>
      <th>ProofType_Pos</th>
      <th>ProofType_Proof of Authority</th>
      <th>ProofType_Proof of Trust</th>
      <th>ProofType_TPoS</th>
      <th>ProofType_Zero-Knowledge Proof</th>
      <th>ProofType_dPoW/PoW</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>4.199995e+01</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>1.055185e+09</td>
      <td>532000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>2.927942e+10</td>
      <td>314159265359</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>1.792718e+07</td>
      <td>21000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>1.076842e+08</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZEPH</th>
      <td>2.000000e+09</td>
      <td>2000000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GAP</th>
      <td>1.493105e+07</td>
      <td>250000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BDX</th>
      <td>9.802226e+08</td>
      <td>1400222610</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ZEN</th>
      <td>7.296538e+06</td>
      <td>21000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>XBC</th>
      <td>1.283270e+05</td>
      <td>1000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>532 rows × 98 columns</p>
</div>




```python
# Standardize the data with StandardScaler().
crypto_scaled = StandardScaler().fit_transform(X)
print(crypto_scaled[0:5])
```

    [[-0.11710817 -0.1528703  -0.0433963  -0.0433963  -0.0433963  -0.06142951
      -0.07530656 -0.0433963  -0.06142951 -0.06142951 -0.0433963  -0.0433963
      -0.19245009 -0.06142951 -0.09740465 -0.0433963  -0.11547005 -0.07530656
      -0.0433963  -0.0433963  -0.15191091 -0.0433963  -0.13118084 -0.0433963
      -0.0433963  -0.08703883 -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.06142951 -0.0433963  -0.08703883 -0.08703883 -0.08703883 -0.0433963
      -0.13118084 -0.13840913 -0.13840913 -0.0433963  -0.06142951 -0.0433963
      -0.07530656 -0.18168574 -0.0433963  -0.0433963  -0.0433963  -0.07530656
      -0.15826614 -0.31491833 -0.0433963  -0.08703883 -0.07530656 -0.06142951
       1.38675049 -0.0433963  -0.0433963  -0.06142951 -0.0433963  -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.39879994 -0.0433963  -0.18168574 -0.0433963  -0.08703883 -0.08703883
      -0.10680283 -0.0433963  -0.13118084 -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.07530656 -0.43911856 -0.0433963  -0.06142951 -0.0433963
      -0.0433963  -0.89632016 -0.0433963  -0.0433963   1.42222617 -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.0433963 ]
     [-0.09396955 -0.145009   -0.0433963  -0.0433963  -0.0433963  -0.06142951
      -0.07530656 -0.0433963  -0.06142951 -0.06142951 -0.0433963  -0.0433963
      -0.19245009 -0.06142951 -0.09740465 -0.0433963  -0.11547005 -0.07530656
      -0.0433963  -0.0433963  -0.15191091 -0.0433963  -0.13118084 -0.0433963
      -0.0433963  -0.08703883 -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.06142951 -0.0433963  -0.08703883 -0.08703883 -0.08703883 -0.0433963
      -0.13118084 -0.13840913 -0.13840913 -0.0433963  -0.06142951 -0.0433963
      -0.07530656 -0.18168574 -0.0433963  -0.0433963  -0.0433963  -0.07530656
      -0.15826614 -0.31491833 -0.0433963  -0.08703883 -0.07530656 -0.06142951
       1.38675049 -0.0433963  -0.0433963  -0.06142951 -0.0433963  -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.39879994 -0.0433963  -0.18168574 -0.0433963  -0.08703883 -0.08703883
      -0.10680283 -0.0433963  -0.13118084 -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.07530656 -0.43911856 -0.0433963  -0.06142951 -0.0433963
      -0.0433963  -0.89632016 -0.0433963  -0.0433963   1.42222617 -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.0433963 ]
     [ 0.52494561  4.48942416 -0.0433963  -0.0433963  -0.0433963  -0.06142951
      -0.07530656 -0.0433963  -0.06142951 -0.06142951 -0.0433963  -0.0433963
      -0.19245009 -0.06142951 -0.09740465 -0.0433963  -0.11547005 -0.07530656
      -0.0433963  -0.0433963  -0.15191091 -0.0433963  -0.13118084 -0.0433963
      -0.0433963  -0.08703883 -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.06142951 -0.0433963  -0.08703883 -0.08703883 -0.08703883 -0.0433963
      -0.13118084 -0.13840913 -0.13840913 -0.0433963  -0.06142951 -0.0433963
      -0.07530656 -0.18168574 -0.0433963  -0.0433963  -0.0433963  -0.07530656
      -0.15826614 -0.31491833 -0.0433963  -0.08703883 -0.07530656 -0.06142951
      -0.72111026 -0.0433963  -0.0433963  -0.06142951 -0.0433963  -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.39879994 -0.0433963   5.50400923 -0.0433963  -0.08703883 -0.08703883
      -0.10680283 -0.0433963  -0.13118084 -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.07530656 -0.43911856 -0.0433963  -0.06142951 -0.0433963
      -0.0433963  -0.89632016 -0.0433963  -0.0433963   1.42222617 -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.0433963 ]
     [-0.11671506 -0.15255998 -0.0433963  -0.0433963  -0.0433963  -0.06142951
      -0.07530656 -0.0433963  -0.06142951 -0.06142951 -0.0433963  -0.0433963
      -0.19245009 -0.06142951 -0.09740465 -0.0433963  -0.11547005 -0.07530656
      -0.0433963  -0.0433963  -0.15191091 -0.0433963  -0.13118084 -0.0433963
      -0.0433963  -0.08703883 -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.06142951 -0.0433963  -0.08703883 -0.08703883 -0.08703883 -0.0433963
      -0.13118084 -0.13840913 -0.13840913 -0.0433963  -0.06142951 -0.0433963
      -0.07530656 -0.18168574 -0.0433963  -0.0433963  -0.0433963  -0.07530656
      -0.15826614  3.17542648 -0.0433963  -0.08703883 -0.07530656 -0.06142951
      -0.72111026 -0.0433963  -0.0433963  -0.06142951 -0.0433963  -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.39879994 -0.0433963  -0.18168574 -0.0433963  -0.08703883 -0.08703883
      -0.10680283 -0.0433963  -0.13118084 -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.07530656 -0.43911856 -0.0433963  -0.06142951 -0.0433963
      -0.0433963   1.11567277 -0.0433963  -0.0433963  -0.70312305 -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.0433963 ]
     [-0.11474682 -0.1528703  -0.0433963  -0.0433963  -0.0433963  -0.06142951
      -0.07530656 -0.0433963  -0.06142951 -0.06142951 -0.0433963  -0.0433963
      -0.19245009 -0.06142951 -0.09740465 -0.0433963  -0.11547005 -0.07530656
      -0.0433963  -0.0433963  -0.15191091 -0.0433963   7.62306442 -0.0433963
      -0.0433963  -0.08703883 -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.06142951 -0.0433963  -0.08703883 -0.08703883 -0.08703883 -0.0433963
      -0.13118084 -0.13840913 -0.13840913 -0.0433963  -0.06142951 -0.0433963
      -0.07530656 -0.18168574 -0.0433963  -0.0433963  -0.0433963  -0.07530656
      -0.15826614 -0.31491833 -0.0433963  -0.08703883 -0.07530656 -0.06142951
      -0.72111026 -0.0433963  -0.0433963  -0.06142951 -0.0433963  -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.39879994 -0.0433963  -0.18168574 -0.0433963  -0.08703883 -0.08703883
      -0.10680283 -0.0433963  -0.13118084 -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.07530656 -0.43911856 -0.0433963  -0.06142951 -0.0433963
      -0.0433963   1.11567277 -0.0433963  -0.0433963  -0.70312305 -0.0433963
      -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963  -0.0433963
      -0.0433963  -0.0433963 ]]


### Deliverable 2: Reducing Data Dimensions Using PCA


```python
# Using PCA to reduce dimension to three principal components.
pca = PCA(n_components=3)

crypto_pca = pca.fit_transform(crypto_scaled)
```


```python
# Create a DataFrame with the three principal components.

df_crypto_pca = pd.DataFrame(
    data=crypto_pca, columns=["PC 1", "PC 2", "PC 3"], index = crypto_df.index
)
df_crypto_pca.head()

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC 1</th>
      <th>PC 2</th>
      <th>PC 3</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>-0.336814</td>
      <td>1.031056</td>
      <td>-0.573032</td>
    </tr>
    <tr>
      <th>404</th>
      <td>-0.320144</td>
      <td>1.031300</td>
      <td>-0.573084</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>2.309937</td>
      <td>1.696278</td>
      <td>-0.561902</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>-0.144440</td>
      <td>-1.329395</td>
      <td>0.163981</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>-0.151435</td>
      <td>-2.050480</td>
      <td>0.367377</td>
    </tr>
  </tbody>
</table>
</div>



### Deliverable 3: Clustering Crytocurrencies Using K-Means

#### Finding the Best Value for `k` Using the Elbow Curve


```python
# Create an elbow curve to find the best value for K.
sse = {}
K = range(1,10)
for k in K:
    kmeanmodel = KMeans(n_clusters=k).fit(df_crypto_pca)
    sse[k]= kmeanmodel.inertia_
    
# Plot
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

```


    
![png](output_20_0.png)
    


Running K-Means with `k=4`


```python
# Initialize the K-Means model
model = KMeans(n_clusters=4, random_state=0)

# Fit the model
model.fit(df_crypto_pca)

# Predict clusters
predictions = model.predict(df_crypto_pca)

# Add the predicted class columns
df_crypto_pca["class"] = model.labels_
df_crypto_pca.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC 1</th>
      <th>PC 2</th>
      <th>PC 3</th>
      <th>class</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>-0.336814</td>
      <td>1.031056</td>
      <td>-0.573032</td>
      <td>0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>-0.320144</td>
      <td>1.031300</td>
      <td>-0.573084</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>2.309937</td>
      <td>1.696278</td>
      <td>-0.561902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>-0.144440</td>
      <td>-1.329395</td>
      <td>0.163981</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>-0.151435</td>
      <td>-2.050480</td>
      <td>0.367377</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a new DataFrame including predicted clusters and cryptocurrencies features.
# Concatentate the crypto_df and pcs_df DataFrames on the same columns.

clustered_df = pd.concat([crypto_df,df_crypto_pca],axis =1)

#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
clustered_df['CoinName'] = crypto_names['CoinName']

# Add the predicted class columns
clustered_df["class"] = model.labels_

# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)
```

    (532, 9)





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>ProofType</th>
      <th>TotalCoinsMined</th>
      <th>TotalCoinSupply</th>
      <th>PC 1</th>
      <th>PC 2</th>
      <th>PC 3</th>
      <th>class</th>
      <th>CoinName</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>4.199995e+01</td>
      <td>42</td>
      <td>-0.336814</td>
      <td>1.031056</td>
      <td>-0.573032</td>
      <td>0</td>
      <td>42 Coin</td>
    </tr>
    <tr>
      <th>404</th>
      <td>Scrypt</td>
      <td>PoW/PoS</td>
      <td>1.055185e+09</td>
      <td>532000000</td>
      <td>-0.320144</td>
      <td>1.031300</td>
      <td>-0.573084</td>
      <td>0</td>
      <td>404Coin</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>X13</td>
      <td>PoW/PoS</td>
      <td>2.927942e+10</td>
      <td>314159265359</td>
      <td>2.309937</td>
      <td>1.696278</td>
      <td>-0.561902</td>
      <td>0</td>
      <td>EliteCoin</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>SHA-256</td>
      <td>PoW</td>
      <td>1.792718e+07</td>
      <td>21000000</td>
      <td>-0.144440</td>
      <td>-1.329395</td>
      <td>0.163981</td>
      <td>3</td>
      <td>Bitcoin</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.076842e+08</td>
      <td>0</td>
      <td>-0.151435</td>
      <td>-2.050480</td>
      <td>0.367377</td>
      <td>3</td>
      <td>Ethereum</td>
    </tr>
    <tr>
      <th>LTC</th>
      <td>Scrypt</td>
      <td>PoW</td>
      <td>6.303924e+07</td>
      <td>84000000</td>
      <td>-0.169932</td>
      <td>-1.127602</td>
      <td>-0.039960</td>
      <td>3</td>
      <td>Litecoin</td>
    </tr>
    <tr>
      <th>DASH</th>
      <td>X11</td>
      <td>PoW/PoS</td>
      <td>9.031294e+06</td>
      <td>22000000</td>
      <td>-0.383913</td>
      <td>1.211396</td>
      <td>-0.429342</td>
      <td>0</td>
      <td>Dash</td>
    </tr>
    <tr>
      <th>XMR</th>
      <td>CryptoNight-V7</td>
      <td>PoW</td>
      <td>1.720114e+07</td>
      <td>0</td>
      <td>-0.151803</td>
      <td>-2.227887</td>
      <td>0.416721</td>
      <td>3</td>
      <td>Monero</td>
    </tr>
    <tr>
      <th>ETC</th>
      <td>Ethash</td>
      <td>PoW</td>
      <td>1.133597e+08</td>
      <td>210000000</td>
      <td>-0.149876</td>
      <td>-2.050568</td>
      <td>0.367393</td>
      <td>3</td>
      <td>Ethereum Classic</td>
    </tr>
    <tr>
      <th>ZEC</th>
      <td>Equihash</td>
      <td>PoW</td>
      <td>7.383056e+06</td>
      <td>21000000</td>
      <td>-0.157888</td>
      <td>-1.984972</td>
      <td>0.329870</td>
      <td>3</td>
      <td>ZCash</td>
    </tr>
  </tbody>
</table>
</div>



### Deliverable 4: Visualizing Cryptocurrencies Results

#### 3D-Scatter with Clusters


```python
# Creating a 3D-Scatter with the PCA data and the clusters
import plotly.express as px

fig = px.scatter_3d(
  clustered_df,
  hover_name="CoinName",
  hover_data=["Algorithm"],
  x="PC 1",
  y="PC 2",
  z="PC 3",
  color="class",
  symbol="class",
  width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()



```




```python
# Create a table with tradable cryptocurrencies.
clustered_df.hvplot.table(sortable=True, selectable=True)
```










```python
# Print the total number of tradable cryptocurrencies.
clustered_df['CoinName'].count()
```




    532




```python
# Scaling data to create the scatter plot with tradable cryptocurrencies.
scaled =MinMaxScaler().fit_transform(clustered_df[["TotalCoinSupply","TotalCoinsMined"]])
print(scaled)
```

    [[4.20000000e-11 0.00000000e+00]
     [5.32000000e-04 1.06585544e-03]
     [3.14159265e-01 2.95755135e-02]
     ...
     [1.40022261e-03 9.90135079e-04]
     [2.10000000e-05 7.37028150e-06]
     [1.00000000e-06 1.29582282e-07]]



```python
# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
final_df = pd.DataFrame(
    data = scaled,columns = ["TotalCoinSupply","TotalCoinsMined"], index = clustered_df.index
)

# Add the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
final_df = pd.concat([final_df,clustered_df['CoinName']],axis =1)

# Add the "Class" column from the clustered_df DataFrame to the new DataFrame. 
plot_df = pd.concat([final_df,clustered_df["class"]],axis =1)

plot_df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalCoinSupply</th>
      <th>TotalCoinsMined</th>
      <th>CoinName</th>
      <th>class</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>4.200000e-11</td>
      <td>0.000000</td>
      <td>42 Coin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>404</th>
      <td>5.320000e-04</td>
      <td>0.001066</td>
      <td>404Coin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>3.141593e-01</td>
      <td>0.029576</td>
      <td>EliteCoin</td>
      <td>0</td>
    </tr>
    <tr>
      <th>BTC</th>
      <td>2.100000e-05</td>
      <td>0.000018</td>
      <td>Bitcoin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ETH</th>
      <td>0.000000e+00</td>
      <td>0.000109</td>
      <td>Ethereum</td>
      <td>3</td>
    </tr>
    <tr>
      <th>LTC</th>
      <td>8.400000e-05</td>
      <td>0.000064</td>
      <td>Litecoin</td>
      <td>3</td>
    </tr>
    <tr>
      <th>DASH</th>
      <td>2.200000e-05</td>
      <td>0.000009</td>
      <td>Dash</td>
      <td>0</td>
    </tr>
    <tr>
      <th>XMR</th>
      <td>0.000000e+00</td>
      <td>0.000017</td>
      <td>Monero</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ETC</th>
      <td>2.100000e-04</td>
      <td>0.000115</td>
      <td>Ethereum Classic</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ZEC</th>
      <td>2.100000e-05</td>
      <td>0.000007</td>
      <td>ZCash</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
plot_df.hvplot.scatter(x="TotalCoinsMined", y="TotalCoinSupply", by="class",hover_cols = ["CoinName"])

```




```python

```
