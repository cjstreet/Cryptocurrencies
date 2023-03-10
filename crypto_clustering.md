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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>532 rows ?? 98 columns</p>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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


<div>                            <div id="58a5fc45-af4a-4771-a039-dfd908067cf2" class="plotly-graph-div" style="height:525px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("58a5fc45-af4a-4771-a039-dfd908067cf2")) {                    Plotly.newPlot(                        "58a5fc45-af4a-4771-a039-dfd908067cf2",                        [{"customdata":[["Scrypt"],["Scrypt"],["X13"],["X11"],["SHA-512"],["SHA-256"],["SHA-256"],["X15"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Groestl"],["PoS"],["Scrypt"],["Scrypt"],["X11"],["X11"],["SHA3"],["Scrypt"],["SHA-256"],["Scrypt"],["X13"],["X13"],["NeoScrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["X11"],["X11"],["Multiple"],["PHI1612"],["X11"],["Scrypt"],["Scrypt"],["Scrypt"],["X11"],["Multiple"],["X13"],["Scrypt"],["Shabal256"],["Counterparty"],["SHA-256"],["Groestl"],["Scrypt"],["X13"],["Scrypt"],["Scrypt"],["X13"],["X11"],["Scrypt"],["X11"],["SHA3"],["QUAIT"],["X11"],["Scrypt"],["X13"],["SHA-256"],["X15"],["BLAKE256"],["SHA-256"],["X11"],["SHA-256"],["NIST5"],["Scrypt"],["Scrypt"],["X11"],["Scrypt"],["SHA-256"],["Scrypt"],["PoS"],["X11"],["SHA-256"],["SHA-256"],["NIST5"],["X11"],["POS 3.0"],["Scrypt"],["Scrypt"],["Scrypt"],["X13"],["X11"],["X11"],["Scrypt"],["SHA-256"],["X11"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["SHA-256D"],["PoS"],["Scrypt"],["X11"],["PoS"],["X13"],["X14"],["PoS"],["SHA-256D"],["DPoS"],["X11"],["X13"],["X11"],["PoS"],["Scrypt"],["Scrypt"],["PoS"],["X11"],["SHA-256"],["Scrypt"],["X11"],["Scrypt"],["Scrypt"],["X11"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Quark"],["QuBit"],["Scrypt"],["SHA-256"],["X11"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["X13"],["Scrypt"],["Scrypt"],["X11"],["Blake2S"],["X11"],["PoS"],["X11"],["PoS"],["X11"],["Scrypt"],["Scrypt"],["Scrypt"],["SHA-256"],["X11"],["Scrypt"],["PoS"],["Scrypt"],["X15"],["SHA-256"],["POS 3.0"],["536"],["NIST5"],["NIST5"],["Skein"],["X13"],["Scrypt"],["X13"],["SkunkHash v2 Raptor"],["Skein"],["X11"],["Scrypt"],["VeChainThor Authority"],["PoS"],["Scrypt"],["Scrypt"],["SHA-512"],["Ouroboros"],["X11"],["NeoScrypt"],["Scrypt"],["Lyra2REv2"],["Scrypt"],["SHA-256"],["NIST5"],["PHI1612"],["Scrypt"],["Quark"],["POS 2.0"],["Scrypt"],["SHA-256"],["X11"],["DPoS"],["NIST5"],["X13"],["Scrypt"],["NIST5"],["Quark"],["Scrypt"],["Scrypt"],["X11"],["Quark"],["Scrypt"],["Scrypt"],["X11"],["POS 3.0"],["Ethash"],["Scrypt"],["Scrypt"],["X13"],["C11"],["X11"],["XEVAN"],["Scrypt"],["VBFT"],["NIST5"],["Scrypt"],["Scrypt"],["Scrypt"],["Green Protocol"],["PoS"],["Scrypt"],["Semux BFT consensus"],["Quark"],["PoS"],["X16R"],["Scrypt"],["XEVAN"],["Scrypt"],["Scrypt"],["Scrypt"],["SHA-256D"],["Scrypt"],["X15"],["Scrypt"],["Quark"],["SHA-256"],["DPoS"],["X16R"],["Quark"],["Quark"],["Scrypt"],["Lyra2REv2"],["Quark"],["Scrypt"],["X11"],["X11"],["Scrypt"],["PoS"],["Keccak"],["X11"],["Scrypt"],["SHA-512"],["XEVAN"],["XEVAN"],["X11"],["Quark"],["Equihash"],["Scrypt"],["Quark"],["Quark"],["Scrypt"],["X11"],["Scrypt"],["XEVAN"],["SHA-256D"],["X11"],["X11"],["DPoS"],["Scrypt"],["X11"],["Scrypt"],["Scrypt"],["SHA-256"],["Scrypt"],["X11"],["Scrypt"],["SHA-256"],["X11"],["Scrypt"],["Scrypt"],["X11"],["Scrypt"],["PoS"],["X11"],["SHA-256"],["DPoS"],["Scrypt"],["Scrypt"],["NeoScrypt"],["X13"],["DPoS"],["DPoS"],["SHA-256"],["PoS"],["PoS"],["SHA-256"],["Scrypt"],["Scrypt"]],"hovertemplate":"<b>%{hovertext}</b><br><br>class=%{marker.color}<br>PC 1=%{x}<br>PC 2=%{y}<br>PC 3=%{z}<br>Algorithm=%{customdata[0]}<extra></extra>","hovertext":["42 Coin","404Coin","EliteCoin","Dash","Bitshares","BitcoinDark","PayCoin","KoboCoin","Aurora Coin","BlueCoin","EnergyCoin","BitBar","CryptoBullion","CasinoCoin","Diamond","Exclusive Coin","FlutterCoin","HoboNickels","HyperStake","IOCoin","MaxCoin","MintCoin","MazaCoin","Nautilus Coin","NavCoin","OpalCoin","Orbitcoin","PotCoin","PhoenixCoin","Reddcoin","SuperCoin","SyncCoin","TeslaCoin","TittieCoin","TorCoin","UnitaryStatus Dollar","UltraCoin","VeriCoin","X11 Coin","Crypti","StealthCoin","ZCC Coin","BurstCoin","StorjCoin","Neutron","FairCoin","RubyCoin","Kore","Dnotes","8BIT Coin","Sativa Coin","Ucoin","Vtorrent","IslaCoin","Nexus","Droidz","Squall Coin","Diggits","Paycon","Emercoin","EverGreenCoin","Decred","EDRCoin","Hitcoin","DubaiCoin","PWR Coin","BillaryCoin","GPU Coin","EuropeCoin","ZeitCoin","SwingCoin","SafeExchangeCoin","Nebuchadnezzar","Ratecoin","Revenu","Clockcoin","VIP Tokens","BitSend","Let it Ride","PutinCoin","iBankCoin","Frankywillcoin","MudraCoin","Lutetium Coin","GoldBlocks","CarterCoin","BitTokens","MustangCoin","ZoneCoin","RootCoin","BitCurrency","Swiscoin","BuzzCoin","Opair","PesoBit","Halloween Coin","CoffeeCoin","RoyalCoin","GanjaCoin V2","TeamUP","LanaCoin","ARK","InsaneCoin","EmberCoin","XenixCoin","FreeCoin","PLNCoin","AquariusCoin","Creatio","Eternity","Eurocoin","BitcoinFast","Stakenet","BitConnect Coin","MoneyCoin","Enigma","Russiacoin","PandaCoin","GameUnits","GAKHcoin","Allsafe","LiteCreed","Klingon Empire Darsek","Internet of People","KushCoin","Printerium","Impeach","Zilbercoin","FirstCoin","FindCoin","OpenChat","RenosCoin","VirtacoinPlus","TajCoin","Impact","Atmos","HappyCoin","MacronCoin","Condensate","Independent Money System","ArgusCoin","LomoCoin","ProCurrency","GoldReserve","GrowthCoin","Phreak","Degas Coin","HTML5 Coin","Ultimate Secure Cash","QTUM","Espers","Denarius","Virta Unique Coin","Bitcoin Planet","BritCoin","Linda","DeepOnion","Signatum","Cream","Monoeci","Draftcoin","Vechain","Stakecoin","CoinonatX","Ethereum Dark","Obsidian","Cardano","Regalcoin","TrezarCoin","TerraNovaCoin","Rupee","WomenCoin","Theresa May Coin","NamoCoin","LUXCoin","Xios","Bitcloud 2.0","KekCoin","BlackholeCoin","Infinity Economics","Magnet","Lamden Tau","Electra","Bitcoin Diamond","Cash & Back Coin","Bulwark","Kalkulus","GermanCoin","LiteCoin Ultra","PhantomX","Digiwage","Trollcoin","Litecoin Plus","Monkey Project","TokenPay","1717 Masonic Commemorative Token","My Big Coin","Unified Society USDEX","Tokyo Coin","Stipend","Pushi","Ellerium","Velox","Ontology","Bitspace","Briacoin","Ignition","MedicCoin","Bitcoin Green","Deviant Coin","Abjcoin","Semux","Carebit","Zealium","Proton","iDealCash","Bitcoin Incognito","HollyWoodCoin","Swisscoin","Xt3ch","TheVig","EmaratCoin","Dekado","Lynx","Poseidon Quark","BitcoinWSpectrum","Muse","Trivechain","Dystem","Giant","Peony Coin","Absolute Coin","Vitae","TPCash","ARENON","EUNO","MMOCoin","Ketan","XDNA","PAXEX","ThunderStake","Kcash","Bettex coin","BitMoney","Junson Ming Chan Coin","HerbCoin","PirateCash","Oduwa","Galilel","Crypto Sports","Credit","Dash Platinum","Nasdacoin","Beetle Coin","Titan Coin","Award","Insane Coin","ALAX","LiteDoge","TruckCoin","OrangeCoin","BitstarCoin","NeosCoin","HyperCoin","PinkCoin","AudioCoin","IncaKoin","Piggy Coin","Genstake","XiaoMiCoin","CapriCoin"," ClubCoin","Radium","Creditbit ","OKCash","Lisk","HiCoin","WhiteCoin","FriendshipCoin","Triangles Coin","EOS","Oxycoin","TigerCash","Particl","Nxt","ZEPHYR","Gapcoin","BitcoinPlus"],"legendgroup":"0","marker":{"color":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"coloraxis":"coloraxis","symbol":"circle"},"mode":"markers","name":"0","scene":"scene","showlegend":true,"x":[-0.336814160243832,-0.3201436850611277,2.309936601848046,-0.38391286709034184,-0.2784818967802449,-0.31015224161744126,-0.2691789537546637,-0.23615573718861377,-0.33647355965858694,-0.32904792312712416,-0.33531007064608165,-0.33681009378048077,-0.3367943671042389,0.6138136769432345,-0.3834372805866841,-0.35673647695940613,-0.2873868820633119,-0.33487188138617285,-0.3227543860575429,-0.38380751691913717,-0.46096704631933627,-0.03564706575717692,-0.27343839125958636,-0.29559180814054614,-0.2756721930283528,-0.27629467540838776,-0.3412180696135936,-0.33111320507209874,-0.33521503563797544,0.02148471569252018,-0.34264973249618763,-0.38417928927423833,-0.23743064889672993,-0.2942985044870644,-0.3840908732260523,-0.3120326522292737,-0.33549080694123834,-0.30247934812178257,-0.38405534125723184,-0.2378987620646202,-0.27607544252470106,-0.3279116792640565,-0.1196573545413747,-0.3269059338112433,-0.3093635322210691,-0.3828585967158138,-0.29557208621972736,-0.2763700750280736,-0.29022292764421453,-0.3367962209013715,-0.27632224284350254,-0.343094738735234,-0.3365304520568297,-0.38416081005582664,-0.531721135529826,-0.41007842914234244,-0.3166889328005052,-0.2939727526056126,-0.27584360246536754,-0.30270404866748196,-0.23859880215962803,-0.0789359850834064,-0.3101231427890185,-0.061442410617304447,-0.3101868845893858,-0.22963115865473868,-0.3364062407397421,-0.2938415362880781,-0.3813281213401363,0.8582419301064816,-0.3099868099274077,-0.11734026419613088,-0.3564195778560324,-0.38200273247256006,-0.30872949447753206,-0.3063961954052742,-0.35104296625848014,-0.38289521289256856,-0.35648633674787344,-0.31268898546891033,-0.33644435123796157,-0.3348825748764592,-0.2340902406463176,-0.3305789672889262,-0.38363440072436855,-0.3356481511673718,-0.31016781057536896,-0.38414999185743454,-0.336633635642617,-0.33679045146606795,-0.2938314812727351,-0.258247076513355,0.027405351590154295,-0.35537610812829024,-0.33640443888362753,-0.37030378713563344,-0.3547045095297956,-0.23552180424844193,-0.4190428708862358,-0.3544529015427051,-0.28727970364742755,3.7093443455831503,-0.38374231063580894,0.8563430527944841,-0.38410488042597457,-0.35584010200244115,-0.3363318974683111,-0.33648628554138305,-0.3564195778560324,-0.3836795052221467,-0.310030424891535,-0.3363326429061574,-0.34576780189108397,-0.3364801066533051,-0.3320651617270648,-0.38413388510871116,-0.33569026916768874,0.3480164092698958,-0.3366794953583855,-0.3367501137330596,-0.3557633095549336,-0.35644483613570666,-0.33297439057892014,-0.3101442139130774,-0.3840437897932788,-0.3365277997797784,-0.3366548054201363,-0.2954784633301971,-0.25947646255684004,-0.23528953627436752,-0.31749829926872564,-0.2952195147142574,-0.3833090613479351,-0.38968119537603335,-0.38204686127027687,-0.35466521054823286,-0.3832069017249386,-0.3490621778195266,-0.37914543652367544,-0.33659806825892474,-0.33659724555162257,-0.3297143251516174,0.26384401254966683,-0.3836856907775702,-0.2781097030251626,-0.3564538704449447,-0.3358082702087011,0.8964246206497702,-0.26786847673861885,-0.35526404027764935,0.5731914641897556,-0.35257942329731284,-0.35108081010688974,-0.38827167574002003,-0.2760071652907533,0.12842046539903423,-0.2760779872298896,-0.33068332450691806,-0.3877763438122163,-0.38395824554092073,-0.29555276080542625,4.400351759693226,-0.3563200333423979,-0.33623305912384843,-0.33673303443559344,-0.33657735571255354,0.6210640034973482,-0.3838990181105123,-0.3362144542980742,-0.33668861090167845,-0.31535339058892725,0.43280726874379344,-0.3084896262441037,-0.33733759350100007,-0.3708250526111716,-0.33663891905242255,-0.3951139816135867,-0.3630420553000833,-0.3365092718286669,-0.09557147798572486,-0.3826909459197644,3.714203114077511,0.20720870444698564,-0.2727470583661322,-0.33396810302481705,-0.31143300210652936,-0.3556491549088517,0.07790110962447107,-0.33564719924162995,-0.38327249908017175,-0.3548133576784111,-0.3232062477361639,-0.3367551759820883,-0.3430666542235132,-0.3567683912156494,-0.27732573479289685,-0.3026852189433399,-0.3322980936704688,-0.22708037142896947,-0.3642956985399379,-0.3839754721528403,-0.2733257411894085,-0.29471219670232734,-0.34074792805358584,-0.35217814045846685,-0.33678254437942373,-0.3367642439889468,-0.2890905275600743,-0.36671862411214445,-0.3559142686657329,-0.33648058845690976,3.9032828566223574,-0.39378142590411225,-0.35610303383266323,-0.32564578219151075,-0.2833186428557501,-0.20598788452149544,-0.2952468220122891,-0.13979237083411492,-0.29549895998638226,-0.3526359280771732,-0.3359542980814159,-0.19703233014742644,2.2271928582478773,-0.3513541453812955,-0.2679964274756907,1.009377472066068,-0.36589180387865183,-0.35576273489677324,-0.39679919876589315,-0.17614952795049887,-0.3561914199064097,-0.3544815730642903,-0.32965439020593007,-0.342643819027174,-0.3834485355958408,-0.2927428655750498,-0.35520221539666585,-0.3353089801214128,-0.34250505968429695,-0.19617221009180866,-0.33029762563272885,-0.23240902401237068,0.64713404023358,-0.13292744375237456,-0.3960934358458447,-0.28182381183491684,-0.33648457960106115,-0.39654981374938253,-0.3558797757964213,0.5705525058466439,-0.343120995215607,-0.3359671656873473,-0.2677781774578613,-0.3073274690659458,-0.38101147810436015,-0.3836759059651779,3.7264510310760723,0.09869574401671052,-0.38121453681851736,-0.3353521845563782,-0.33617779388420693,-0.2692115829374906,-0.3366964472097747,-0.3752992745781706,-0.2503434030057113,-0.0897144721190863,-0.37104490881127883,-0.3359744269654891,-0.32903505978528425,-0.3802427136007989,-0.3344126843198244,-0.35669534777946715,-0.35271163093885394,-0.3086665140112174,3.709736391226263,-0.10268005287274921,-0.3316059970865271,-0.340843666006022,-0.2764773467169601,3.719608404215658,3.7208530792688363,-0.25009837215446934,-0.3566311870998663,-0.36664019415408977,2.467325835154057,-0.3348582405554161,-0.29589567662929084],"y":[1.0310559715245244,1.0312996888842636,1.6962776903440542,1.2113962542744472,1.4511214328246347,0.8292466263305839,0.6413239603207586,1.9262873108663563,1.0310568815839762,1.0313404832959474,1.0311110730326034,1.0310557756048737,1.0310560068216175,0.4775204424058761,0.9678863630874658,1.6236067101434604,0.35203001854145066,1.031044344757026,1.0242216845296954,1.2114001137189065,2.0356405626808844,0.8526588062626184,0.8289379273441801,0.8431247147184787,1.8182757576453523,1.8182529533523686,0.47370263369995164,1.0309750890228233,1.0310469500272048,1.0441820499851322,1.0234928296676875,1.2114016698910732,0.16946420602666445,1.6695743189040582,1.211398011228816,1.0308600802928523,1.0310354676152365,0.2619764130658022,1.2114024172135758,0.1695160412077446,1.8182609848261444,1.030692266507532,0.8910360361327258,1.6196221796162749,0.8292437875046296,0.9679105843565334,0.8431365988707553,1.8182419130202323,0.8429876410734046,1.0310566287512446,1.8182450450103866,1.023462730264717,1.0310525681949496,1.211402347556722,2.264837625101552,1.8443056068356358,0.7052310627455102,0.8431262053164787,1.8182349860344682,0.828844820791827,1.926421113930124,0.9482931620955107,0.8292476923498482,1.2049096382587128,0.8292532903928506,1.510172765627669,1.0310419420731132,0.8430475051090359,1.211241222020624,0.8171115263646372,0.8292402696637801,0.47684730015095966,1.6236045227155298,1.2114296693356144,0.8291602792495311,0.8290372362276799,1.5056628263082787,1.2113528238993188,1.5187721353201848,1.0305600989144945,1.0310389362842363,1.031057749855911,1.630230867726643,1.0234818088967623,1.2113871401451723,1.0310366018221835,0.8292467458129844,1.2114006743480186,1.0310480982796912,1.0310568401112719,0.8432003649570794,0.30379513561977234,1.8617046099018055,1.6236054981519163,1.031070981465803,1.2108752281542348,1.623653556503529,1.6303146669941198,2.015288481845724,1.6234827250724848,1.8587951908497076,1.724899511150114,1.2113969837749745,1.8591601354078902,1.2114017383189872,1.6236050562151727,1.0310470524520403,1.0310390096766837,1.6236045227155298,1.2113785893427327,0.8292524687029283,1.03105084687049,0.6717744387267712,1.031048893831018,1.0307810956812797,1.2113998847248368,1.030997807360811,0.8442839327797755,1.031051936974278,1.0310560304895127,1.5565772628955532,0.577153042299122,1.0308517181396737,0.8292476102634775,1.2114001817421969,1.0310526653596324,1.031046678884042,0.8431020873583314,0.3533535645983943,1.6303148808337127,1.0310737548451665,0.8431257663223568,1.2113645666485557,1.989324234906673,1.2114039082162669,1.6236065070831467,1.211368309209066,1.6236119168115704,1.2112411617314514,1.0310492549407817,1.0310441886076882,1.0306262275861349,0.6091213974747398,1.2113921595872827,0.8423966385978552,1.6235963680013106,1.0310203883141993,1.905934382050016,0.6412425653682693,1.5187710389434523,1.991759338577051,1.5056617264617327,1.505640744646752,1.9307192367803074,1.8182427908509702,1.0136074589557862,1.818247854830336,1.5973854322276566,1.9307373829877637,1.2114032098750782,0.8431252995239279,2.5416978372875247,1.6235794720513963,1.0310439735895451,1.0310560462137213,1.451413920644685,1.5214067858347908,1.2113933124118363,0.47361260206955386,1.0310497174623403,0.7294199773899391,1.0420045599670056,0.8292537276018718,1.5053991898407664,1.8577463117385968,1.0310479047249352,1.7444383051764611,1.6004632605502485,1.0310569393891675,0.6414840138816952,1.2113568578931142,1.7248188184927482,1.5054808757299964,1.8182380514001517,1.031015368350496,1.3177266282588462,1.556577995674783,1.011756710681793,1.0309952461487326,1.2114003982089994,1.5565396303931827,1.0309336315978785,1.0310553730390688,1.023463069282612,1.5187676660736817,-0.07981002433227795,0.29724514176793726,1.031061371673068,1.6300737648880128,0.950168839418276,1.211391891245752,1.4606637445257347,0.8430825599778946,1.5595037680082324,1.5056488335357343,1.0310550602556223,1.0310543509952763,0.8430291259155819,1.456607521920864,1.623576125148126,1.0310474964955225,1.7316630038773095,1.7444871226077505,1.6235747285816329,0.7434594376682492,1.0294824104853426,0.9545144538008126,0.8431305788390507,1.0312373614019992,0.8431089247417218,1.8615098753608867,1.0310295253563075,1.7384696810423501,0.2659065897084552,1.5563007402002058,0.6413067394138785,1.62369321919351,0.9313893989563691,1.5565731448931608,1.7445109833015613,0.8358670368597174,0.9173344915142753,1.5565655819854047,1.0306284232692662,1.0234551049698188,1.2113939492099468,0.8430608867470347,1.6235180500420838,1.011025465508376,1.0234291454305084,1.0237911465282408,1.3469260474917,1.2727393510017773,1.1935728975864162,1.0311758881303201,1.7445031404168259,-0.01428704537258703,1.0310535588642018,1.7445105409201545,1.5565741206632806,1.012696625723451,1.0234619753260126,1.0310290539587783,1.4605634457367522,1.8597894973716966,1.2112279877400032,1.2113994164728517,1.7249225928422818,1.0228661471329517,1.2115102826248765,1.0309715617240565,1.0310418563044812,0.6413169013081553,1.031060283904496,1.2113820637178883,1.0269804335877204,0.8372063468919779,1.2111929987998138,1.031076387074649,1.0310650175591525,1.211402397594965,1.0310335734862819,1.6236020083058995,0.574259862212316,0.8292437982054731,1.724889785327999,0.8432985707143461,1.0310398169571027,0.47367744393194117,1.818246178510304,1.725361759489002,1.725407357461719,0.6413417473150138,1.6236046111821816,1.8214028588283526,0.7425714931703369,1.030955164963327,0.8431240544750843],"z":[-0.573032350780859,-0.5730839895027243,-0.561901566113461,-0.42934168528428185,-0.3769652502439791,-0.36908846837277526,-0.06712458502920467,-0.6007922441271191,-0.5730326691087406,-0.5730873600678896,-0.573043004451103,-0.5730323171453657,-0.5730323660954487,-0.35683126485504946,-0.40026452177135086,-0.03149369313873792,-0.3547283051955647,-0.5730311216340059,-0.12752233742712293,-0.42934243149343665,-0.8709025646628444,-0.2729102680548698,-0.36904913571529274,-0.2710670244715478,-0.5828221049923136,-0.5828176958675169,-0.260176380755522,-0.5730202784105237,-0.5730314385649081,-0.5755702289801266,-0.12738141606431613,-0.4293425455936513,0.16973709022350802,-0.22643572900818895,-0.42934192310028496,-0.5730080605502242,-0.5730292354642395,-0.49125352706382885,-0.4293427367742595,0.16972791712296764,-0.5828192487229467,-0.5729705201545513,-0.2462207192633232,-0.1269720513853345,-0.3690883100538421,-0.4002691676753143,-0.27106918485362397,-0.5828156631451247,-0.27104462961007125,-0.5730324778528371,-0.5828162517229537,-0.12737576626486408,-0.5730318625670459,-0.4293426766092805,-1.0923161349125614,-0.6819327662240382,0.08167099624113196,-0.2710680243366998,-0.5828146464727401,-0.3690190843650244,-0.6008153660964946,-0.4884065331894332,-0.3690886744836149,-0.42831275511197925,-0.36908965920493864,-0.5457910653318071,-0.5730299948344605,-0.2710538357285777,-0.42931478389491934,-0.2668778819375288,-0.36908739216184183,-0.35637973309153415,-0.031493440015313144,-0.4293485959515443,-0.3690734777188598,-0.3690522542400037,-0.5449198495585772,-0.42933428155799785,-0.07957873747143197,-0.5729534564239329,-0.5730294334873004,-0.5730335436521736,-0.28083636884736823,-0.12738486341872282,-0.42934016083207915,-0.5730293698551183,-0.36908848298387154,-0.429342378571771,-0.5730310068160203,-0.5730325187185084,-0.27108151378803064,-0.4153188307253532,-0.6305924991603661,-0.0314940870927793,-0.5730352528965409,-0.42925349561249854,-0.03150309032163565,-0.2808508942735796,-0.5994502481231709,-0.03147227669297152,-0.6299238950630606,-0.6025492554742569,-0.4293418942535139,-0.5907341634123847,-0.4293425915316485,-0.031493797876856855,-0.5730309535304086,-0.573029427866573,-0.031493440015313144,-0.4293385924701875,-0.3690895809928686,-0.5730316401303286,0.03558736162960156,-0.5730312200655513,-0.5729847289126583,-0.4293422428818396,-0.5730223275715289,-0.27156708155628617,-0.5730316810902331,-0.5730323903334748,-0.19545227677594657,-0.33448518433792945,-0.5729971043418303,-0.3690886501221427,-0.4293423372761477,-0.5730318813534738,-0.5730307403055289,-0.2710629791525821,-0.41044850571180297,-0.28085103771316433,-0.5730442794984708,-0.27106738271428005,-0.4293362208498672,-0.7464563455131282,-0.42934391229870295,-0.03149459028198368,-0.4293369444599504,-0.031498095980311475,-0.4293157571222431,-0.5730312322531061,-0.5730303154224378,-0.5729577517759801,-0.06153501221461857,-0.42934104641865223,-0.27094309705993436,-0.03149194823559864,-0.5730263623896533,-0.5976182439412372,-0.06711044028045637,-0.07957909010131935,-0.6962123767269507,-0.5449189576770075,-0.5449158348632538,-0.6804685255923875,-0.5828159856972451,-0.5700832650750469,-0.5828168705401255,-0.06250863046917615,-0.6804720340937612,-0.42934292405564883,-0.2710671479497765,5.784866289261538,-0.031488949762538736,-0.57303044070242,-0.5730324008809783,-0.37699200738775746,0.015189697863791408,-0.4293411589380376,-0.2601623376193892,-0.5730312751629313,-0.009481419428038927,-0.5753614789104401,-0.36909050362662454,-0.5448783007542988,-0.5284287196527743,-0.5730309693929319,-0.49740573546349015,-0.0946170997005935,-0.5730326634715922,-0.0672318380717062,-0.4293351039675146,-0.6025368377428927,-0.5451386173422731,-0.5828165976104395,-0.5730262832882751,-0.24295290169057887,-0.19545246090796806,-0.5697254298151372,-0.5730218833130251,-0.429342724229156,-0.19544589216213606,-0.5730163381545067,-0.5730322690271004,-0.1273758403030861,-0.07957780119110411,0.13628094497941284,-0.4420862355450555,-0.5730353646470613,-0.2808110878284737,-0.4349498201105604,-0.4293408671801292,-0.2915708942720811,-0.27105978944275455,0.04020701548456968,-0.5449168044931408,-0.573032200061211,-0.5730320799091332,-0.2710526505492189,-0.09583952070014132,-0.031488526797616184,-0.5730309668767157,-0.7346742890712405,-0.4974151741313301,-0.031488188853670857,-0.020200659113549825,-0.5727715959453246,0.21943885722876805,-0.2710682416528638,-0.5731540237050684,-0.2710642077409262,-0.6303858894148546,-0.5730279507046196,-0.2988465422973685,-0.37582236616653464,-0.19540420353747626,-0.06712200056565581,-0.35882642160300915,-0.322166191097875,-0.1954515315178551,-0.4974181331653418,-0.26980695981967673,-0.3114438879522343,-0.1954507399978865,-0.5729581763031051,-0.1273745891058809,-0.42934147733929817,-0.2710567536961896,-0.0314783340192591,-0.4243501665347324,-0.12736995199307566,-0.5717805508555804,-0.8499317883162991,0.010393327325077743,0.28981509476887474,-0.1288669066955989,-0.4974170315194416,0.09876969293066852,-0.5730320625996215,-0.4974181655214601,-0.19545165539776518,-0.5701177202837231,-0.1273756177531743,-0.5730278595616697,-0.2915552376431247,-0.630094863621184,-0.42931253074989456,-0.42934236460655373,-0.602561147315122,-0.5717460421303026,-0.4293635454512019,-0.5730177285370257,-0.5730300823105161,-0.0671232923637324,-0.5730331845630542,-0.4293429999945705,-0.5723335097401634,-0.37062887697031455,-0.42931069018916307,-0.5730364254064939,-0.5730374959492447,-0.4293444522829025,-0.5730293786642506,-0.03149286046943797,-0.03973067533692524,-0.3690886262668481,-0.6025476714933742,-0.2711854803159395,-0.5730317744691091,-0.26017198924969537,-0.5828163869959767,-0.6026375681650804,-0.6026463843626416,-0.06713640832770505,-0.03149336061954557,0.16871718993905233,-0.31087135952595546,-0.573014982803607,-0.2710667679319962],"type":"scatter3d"},{"customdata":[["SHA-256"],["Ethash"],["Scrypt"],["CryptoNight-V7"],["Ethash"],["Equihash"],["Multiple"],["Scrypt"],["X11"],["Scrypt"],["Multiple"],["Scrypt"],["SHA-256"],["Scrypt"],["Scrypt"],["Quark"],["Groestl"],["Scrypt"],["Scrypt"],["Scrypt"],["X11"],["Multiple"],["SHA-256"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["NeoScrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["SHA-256"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["HybridScryptHash256"],["Scrypt"],["Scrypt"],["SHA-256"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["SHA-256"],["SHA-256"],["SHA-256"],["SHA-256"],["SHA-256"],["X11"],["Scrypt"],["Lyra2REv2"],["Scrypt"],["SHA-256"],["CryptoNight"],["CryptoNight"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Stanford Folding"],["Multiple"],["QuBit"],["Scrypt"],["Scrypt"],["M7 POW"],["Scrypt"],["SHA-256"],["Scrypt"],["X11"],["Lyra2RE"],["SHA-256"],["X11"],["Scrypt"],["Scrypt"],["Ethash"],["Blake2b"],["X11"],["SHA-256"],["Scrypt"],["1GB AES Pattern Search"],["Scrypt"],["SHA-256"],["X11"],["Dagger"],["Scrypt"],["X11GOST"],["Scrypt"],["X11"],["Scrypt"],["X11"],["Equihash"],["CryptoNight"],["SHA-256"],["Multiple"],["Scrypt"],["SHA-256"],["Scrypt"],["Lyra2Z"],["Ethash"],["Equihash"],["Scrypt"],["X11"],["X11"],["CryptoNight"],["Scrypt"],["CryptoNight"],["Lyra2RE"],["X11"],["CryptoNight-V7"],["Scrypt"],["X11"],["Equihash"],["Scrypt"],["Lyra2RE"],["Dagger-Hashimoto"],["Scrypt"],["NIST5"],["Scrypt"],["SHA-256"],["Scrypt"],["CryptoNight-V7"],["Argon2d"],["Blake2b"],["Cloverhash"],["CryptoNight"],["X11"],["Scrypt"],["Scrypt"],["X11"],["X11"],["CryptoNight"],["Time Travel"],["Scrypt"],["Keccak"],["X11"],["SHA-256"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["Scrypt"],["CryptoNight"],["Equihash"],["X11"],["NeoScrypt"],["Equihash"],["Dagger"],["Scrypt"],["X11"],["NeoScrypt"],["Ethash"],["NeoScrypt"],["Multiple"],["CryptoNight"],["CryptoNight"],["Ethash"],["X11"],["CryptoNight-V7"],["Scrypt"],["BLAKE256"],["X11"],["NeoScrypt"],["NeoScrypt"],["Scrypt"],["X11"],["SHA-256"],["C11"],["CryptoNight"],["SkunkHash"],["CryptoNight"],["Scrypt"],["Dagger"],["Lyra2REv2"],["Scrypt"],["Scrypt"],["X11"],["Ethash"],["CryptoNight"],["Scrypt"],["IMesh"],["Equihash"],["Lyra2Z"],["X11"],["CryptoNight"],["NIST5"],["Lyra2RE"],["Tribus"],["Lyra2Z"],["CryptoNight"],["CryptoNight Heavy"],["CryptoNight"],["Jump Consistent Hash"],["CryptoNight"],["X16R"],["HMQ1725"],["X11"],["Scrypt"],["CryptoNight-V7"],["Cryptonight-GPU"],["XEVAN"],["CryptoNight Heavy"],["SHA-256"],["X11"],["X16R"],["Equihash"],["Lyra2Z"],["SHA-256"],["CryptoNight"],["Blake"],["Blake"],["Exosis"],["Scrypt"],["Equihash"],["Equihash"],["QuBit"],["SHA-256"],["X13"],["SHA-256"],["Scrypt"],["NeoScrypt"],["Blake"],["Scrypt"],["SHA-256"],["Scrypt"],["Groestl"],["Scrypt"],["Scrypt"],["Multiple"],["Equihash+Scrypt"],["Ethash"],["CryptoNight"],["Equihash"]],"hovertemplate":"<b>%{hovertext}</b><br><br>class=%{marker.color}<br>PC 1=%{x}<br>PC 2=%{y}<br>PC 3=%{z}<br>Algorithm=%{customdata[0]}<extra></extra>","hovertext":["Bitcoin","Ethereum","Litecoin","Monero","Ethereum Classic","ZCash","DigiByte","ProsperCoin","Spreadcoin","Argentum","MyriadCoin","MoonCoin","ZetaCoin","SexCoin","Quatloo","QuarkCoin","Riecoin","Digitalcoin ","Catcoin","CannaCoin","CryptCoin","Verge","DevCoin","EarthCoin","E-Gulden","Einsteinium","Emerald","Franko","FeatherCoin","GrandCoin","GlobalCoin","GoldCoin","Infinite Coin","IXcoin","KrugerCoin","LuckyCoin","Litebar ","MegaCoin","MediterraneanCoin","MinCoin","NobleCoin","Namecoin","NyanCoin","RonPaulCoin","StableCoin","SmartCoin","SysCoin","TigerCoin","TerraCoin","UnbreakableCoin","Unobtanium","UroCoin","ViaCoin","Vertcoin","WorldCoin","JouleCoin","ByteCoin","DigitalNote ","MonaCoin","Gulden","PesetaCoin","Wild Beast Coin","Flo","ArtByte","Folding Coin","Unitus","CypherPunkCoin","OmniCron","GreenCoin","Cryptonite","MasterCoin","SoonCoin","1Credit","MarsCoin ","Crypto","Anarchists Prime","BowsCoin","Song Coin","BitZeny","Expanse","Siacoin","MindCoin","I0coin","Revolution VR","HOdlcoin","Gamecredits","CarpeDiemCoin","Adzcoin","SoilCoin","YoCoin","SibCoin","Francs","BolivarCoin","Omni","PizzaCoin","Komodo","Karbo","ZayedCoin","Circuits of Value","DopeCoin","DollarCoin","Shilling","ZCoin","Elementrem","ZClassic","KiloCoin","ArtexCoin","Kurrent","Cannabis Industry Coin","OsmiumCoin","Bikercoins","HexxCoin","PacCoin","Citadel","BeaverCoin","VaultCoin","Zero","Canada eCoin","Zoin","DubaiCoin","EB3coin","Coinonat","BenjiRolls","ILCoin","EquiTrader","Quantum Resistant Ledger","Dynamic","Nano","ChanCoin","Dinastycoin","DigitalPrice","Unify","SocialCoin","ArcticCoin","DAS","LeviarCoin","Bitcore","gCn Coin","SmartCash","Onix","Bitcoin Cash","Sojourn Coin","NewYorkCoin","FrazCoin","Kronecoin","AdCoin","Linx","Sumokoin","BitcoinZ","Elements","VIVO Coin","Bitcoin Gold","Pirl","eBoost","Pura","Innova","Ellaism","GoByte","SHIELD","UltraNote","BitCoal","DaxxCoin","AC3","Lethean","PopularCoin","Photon","Sucre","SparksPay","GunCoin","IrishCoin","Pioneer Coin","UnitedBitcoin","Interzone","TurtleCoin","MUNcoin","Niobio Cash","ShareChain","Travelflex","KREDS","BitFlip","LottoCoin","Crypto Improvement Fund","Callisto Network","BitTube","Poseidon","Aidos Kuneen","Bitrolium","Alpenschillling","FuturoCoin","Monero Classic","Jumpcoin","Infinex","KEYCO","GINcoin","PlatinCoin","Loki","Newton Coin","MassGrid","PluraCoin","Motion","PlusOneCoin","Axe","HexCoin","Webchain","Ryo","Urals Coin","Qwertycoin","Project Pai","Azart","Xchange","CrypticCoin","Actinium","Bitcoin SV","FREDEnergy","Universal Molecule","Lithium","Exosis","Block-Logic","Beam","Bithereum","SLICE","BLAST","Bitcoin Rhodium","GlobalToken","SolarCoin","UFO Coin","BlakeCoin","Crypto Escudo","Crown Coin","SmileyCoin","Groestlcoin","Bata","Pakcoin","JoinCoin","Vollar","Reality Clash","Beldex","Horizen"],"legendgroup":"3","marker":{"color":[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],"coloraxis":"coloraxis","symbol":"diamond"},"mode":"markers","name":"3","scene":"scene","showlegend":true,"x":[-0.1444399535355048,-0.1514346533817148,-0.1699317878373881,-0.1518032419806813,-0.14987560466870806,-0.15788794874043574,0.17386182200972095,-0.1710786656921172,-0.21838513661418235,-0.1706949556954815,-0.07969324764172296,2.5526922027283256,-0.14152328560426364,-0.16794876981570078,-0.17049883452933595,-0.22646756803995802,-0.21679125045713407,-0.1705477138544436,-0.17106092346800159,-0.17114743198444357,-0.21847452378999174,0.19761474808852109,0.23361684886048434,0.07772071113270855,-0.17089359603122653,-0.1664964074899604,-0.1708328434013347,-0.17120446459390526,-0.17083427592598882,-0.15041481465449247,-0.16999891326643612,-0.17027648617107352,1.578667035668819,-0.1444013050118248,-0.16756429895094546,-0.17092005157240936,-0.1712750610135101,-0.1705338995764958,-0.1511427074337487,-0.17115625795306597,-0.03597387603832836,-0.1444789516245356,-0.16481668775075745,-0.17113567160369125,-0.16922874251235417,-0.1706221261165307,-0.13162362613408016,-0.14394242744807262,-0.14422977415102495,-0.14421268829694478,-0.144803801142401,-0.21864852877860988,-0.17085202594262197,-0.18998944232906248,-0.1679534604545277,-0.1440097568330721,3.8613564667240583,0.45845571915483124,-0.16971895797816525,-0.15430809418914743,-0.16843707888257506,-0.17127726923767456,-0.16830206609524706,-0.1545179113448869,-0.13489482238004946,-0.11371894826955802,-0.1917796956004002,-0.17115028663957058,-0.04401267193300517,-0.09173621545794718,-0.17128616923702253,-0.14450674207509304,0.48132562798321077,-0.21803537526696662,-0.14447553159263019,-0.1442464291998594,-0.21832042647073824,-0.16940873272267676,-0.16860053300519023,-0.15250258064556318,0.3694743688624125,-0.21835584794802004,-0.1444024278856283,-0.16724180401331248,-0.16215353651441924,-0.16984871572396057,0.2743117636148153,-0.21751606724278377,-0.1352483231044121,-0.17009611825806442,-0.19004742912569975,-0.171075951180688,-0.2183190594996519,-0.17128622776408362,-0.2184691004756159,-0.29602912482752936,0.303271755256481,-0.14466265257246494,-0.0937801652105864,-0.16845129164271988,-0.14462126148063117,-0.17094942285821052,-0.16057794091127858,-0.15224460212952978,-0.1579095758881596,-0.09796170595358054,-0.20504538142134413,-0.21629590627591086,0.3032661344077639,-0.17126795368223258,0.30339691789595724,-0.14501631923106473,0.4972722672624042,-0.1505670691001921,-0.1712362249630898,-0.2111981836205482,-0.15792196296248961,-0.16936846269482228,-0.14473454998152857,-0.1686428168929708,-0.1421398449228514,-0.18671930413357274,-0.1707983457842126,-0.11096802764521069,-0.17062026855432003,-0.15042486794792082,-0.15407142934317097,-0.03101509577204713,-0.17471904781645128,0.33940821606483274,-0.21752443883849162,-0.17093976428089494,-0.1707029443720411,-0.21791936785276145,-0.21849715601549125,0.30366135980091497,-0.10968863436563275,3.240328564275772,-0.10958782054240827,-0.20936327334266355,-0.1444391173623207,-0.09680809386895779,1.5765489675534328,-0.1710376563614443,-0.17048893998677628,-0.17018974968688297,-0.17017667450665358,0.30384511169466394,0.05453738549729559,-0.17285990783926689,-0.1755257021060264,-0.15776793585192647,-0.1340066355851952,-0.16936667101292932,-0.21403896876977213,-0.1753710127961454,-0.1506086316806143,-0.17547128599968614,-0.10394513661462387,1.141736884126866,0.303248882340799,-0.0754471256768576,-0.21378010034615247,-0.13953040907358588,-0.08737850025404131,1.0924997111566424,-0.21847114102282758,-0.17552326927190753,-0.16855711564600065,-0.170288236541013,-0.2183965084632398,-0.14441850647625687,-0.19874558292748892,8.046312389723075,-0.14553792576807492,0.30712564367201123,0.021860482875256902,-0.13350970947778767,-0.1755391083579243,-0.17086226805630564,0.1363870778576646,-0.21253391359401755,-0.10361521995994015,0.3111773068531886,-0.17109942007660528,-0.16674340809006521,-0.15675847653600386,-0.1583599703866618,-0.21756523038939385,0.3034314960063839,-0.18677883702419137,-0.14486145924065116,-0.11964941128684037,-0.16066237434791558,0.3073624800657769,0.5864732197296482,2.1130578597934306,-0.15161598409374827,0.3170443832139102,-0.20115670548351874,-0.14031986080820086,-0.21845329096025196,-0.1711240116853163,-0.13940903491698164,-0.12658638323785762,-0.10657055133597826,3.1105062338216642,-0.10424472039366224,-0.21842463677682306,-0.20058403566401894,-0.05245273213435375,-0.1600597273025147,-0.14443915494539156,0.3832419081879012,-0.14570390366671848,-0.14609623903950525,-0.19187499379673673,-0.1700156333626595,-0.15582674504304472,-0.15760066053206453,-0.19099013972115356,-0.14371488932262333,-0.11093469220491418,-0.14258213147668056,0.5252753952986704,-0.10064742635028855,-0.0965273422689762,-0.15466856812712476,-0.14422914430659736,0.544967675834248,-0.21635201114274166,-0.17120091333386875,-0.16915538411072642,-0.11445417409917633,-0.15015661417442197,-0.15227777895598357,0.32501841986608504,-0.15788900618444723],"y":[-1.3293952797806805,-2.0504800787365607,-1.1276017057824317,-2.227886853821071,-2.0505678307745834,-1.9849718975261008,-1.8051691517843,-1.12760026093556,-0.9472517231587799,-1.127615867160116,-1.8013509931393632,-1.2927014546096707,-1.329391074798588,-1.1276435728749428,-1.1276335166991838,-0.41412922332987273,-1.1907769510053463,-1.127599550072545,-1.127599610959127,-1.1275973579885037,-0.947253618122958,-1.8012326269857502,-1.3300175854504122,-1.1277840215804318,-1.127593481012297,-1.1276247153464218,-1.1275988436406612,-1.127598133757159,-1.6849980106206737,-1.1278087624570041,-1.1275945070906512,-1.1276062249975645,-1.125984547051184,-1.3293938639136849,-1.1276401259227542,-1.1275937603537696,-1.127593900363996,-1.1275947904257002,-2.1368008326455645,-1.1275955152176382,-1.1329839227039447,-1.3293967084538076,-1.1275888470973046,-1.1276023493151688,-1.127690463957039,-1.1276043691060251,-1.329523854212148,-1.329394997351029,-1.3294020666681234,-1.3294276547503594,-1.3293942949030209,-0.9472475755341412,-1.1275933378002987,-1.2413118949763524,-1.1276543826362462,-1.3293960759843735,-2.3038574438143464,-2.308165758850943,-1.127608478352307,-1.1281303290680622,-1.1276037813176065,-1.1275948628796957,-1.127594429865985,-1.1276689227391345,-2.2284734229259326,-1.801217820714036,-1.5814735314507098,-1.1275907237288232,-1.1298292072580232,-2.1476513219484046,-1.1275938033624537,-1.3293977265413826,-1.167150813696105,-0.9472478777997144,-2.2514123644692092,-1.3294107893636562,-0.9472500423814011,-1.1276696296618547,-1.127667449853964,-2.0505308644033615,-2.2282073957857103,-0.9472478908151352,-1.3293939050495376,-1.1275900798813374,-2.250007337574662,-1.1275986624821286,-1.3289587845915307,-0.9472640352061196,-2.209649665006674,-1.1276659150423076,-2.302170227031932,-1.1275994716485105,-0.9472527516729626,-1.1275938034163373,-0.9472582483399933,-1.1607650301327521,-2.3069593103326635,-1.3293956678497356,-1.8013151855101306,-1.1276274903580608,-1.3293947739537515,-1.1276017347788103,-2.2273600676937733,-2.0505278284479247,-1.9849726898245337,-1.1318055998716328,-0.9470941530802562,-0.9473186726053288,-2.306967104516203,-1.12759458113329,-2.30696507270622,-2.2513936902312857,-0.9900044630960467,-2.2279691881730574,-1.127593864210378,-0.947664478761757,-1.9849703842482753,-1.127592106160795,-2.2513909560403973,-2.220524814765435,-1.1292849865516388,-0.6530018416363967,-1.127600008361144,-1.3298791739643205,-1.1276186497781369,-2.227908791317951,-2.1939313751625913,-2.243113843999344,-2.2943588338656142,-2.307008257817769,-0.947275379373959,-1.1275939836237363,-1.1276237480877003,-0.9472622536791598,-0.9472550681000509,-2.3069753904562114,-2.1483670499413683,-1.1405792721118713,-1.1486154689747765,-0.9476662420970382,-1.3293952491479457,-1.1321082591623504,-1.0635624157468224,-1.1275980687392786,-1.1276221167390403,-1.127622193549276,-1.127621714547379,-2.3069927266444195,-1.9916620191668202,-0.9468118514799451,-1.6849567170728352,-1.98496750092122,-2.2096913082259357,-1.127592040523533,-0.9473201517435181,-1.6849634672807123,-2.0506429737793463,-1.6849580348130262,-1.8013150586178477,-2.3348723198649606,-2.306961872875219,-2.0545947427159863,-0.9474486367330148,-2.228126730241566,-1.1279686778374594,-1.2355766540653834,-0.9472547359135812,-1.6849524888926495,-1.6850277222915824,-1.127600967231928,-0.94725420928743,-1.329393918751091,-1.208482222046511,-2.7131329048520714,-2.2476653201231063,-2.30704301392849,-1.1274159811648816,-2.2096342608424253,-1.241483395017626,-1.1276054403359606,-1.1290198550998327,-0.9473684912577889,-2.053212216498499,-2.307352638903168,-1.1276010212607992,-2.150783313664063,-1.9849643222249758,-2.227471003692414,-0.9472768737495559,-2.306959252996393,-0.6529852230041475,-2.2513992476580613,-2.2070789770718293,-2.2273556415827316,-2.3072164563228843,-2.1329194117298815,-2.367582922596835,-2.284963069515464,-2.3071377018502006,-1.2272472804249444,-2.3016343692266803,-0.947254909797481,-1.1276026846179272,-2.2286400232736194,-2.2130731385593627,-0.6980441216730954,-2.167604636927968,-1.3293569323946215,-0.9472566194373176,-1.2272800564768453,-1.986337653373119,-2.2273842673658395,-1.3293952505247801,-2.3095966771504752,-2.2511949453199054,-2.251154205964558,-2.17941234737465,-1.1276296117442657,-1.985063190387577,-1.9849681926660774,-1.5815135907582836,-1.3293983806849128,-0.3404039977621718,-1.3294286263506414,-1.1697487775281226,-1.6849543374577973,-2.254149770443147,-1.127674441965883,-1.3294020435941265,-1.135845940253727,-1.1907753464106832,-1.1275937019302504,-1.127640867226847,-1.8012466868589303,-2.334911614149873,-2.0505278589924254,-2.307121668032209,-1.9849719362649711],"z":[0.16398133629631523,0.36737744234020864,-0.03996021849194075,0.4167208232493869,0.3673926258890371,0.3298697508351187,0.401422414814378,-0.03995996295376809,0.10372930174260522,-0.03995731063490712,0.4008455053228094,-0.011298391764727654,0.16397925994776236,-0.039953533049088774,-0.039954203816167974,0.03564928103553991,0.1328087375565626,-0.03996033104612127,-0.03996008862428157,-0.039960457493346827,0.10372968510760375,0.4006990421973696,0.16392353710173552,-0.04003887534396395,-0.03996127382658492,-0.03995760183859071,-0.03996033037661192,-0.03996029133410763,0.2729019514776438,-0.03993153318069458,-0.03996149146676162,-0.03995924492011126,-0.04104140526860982,0.16398106254373443,-0.0399543304328509,-0.039961211326576075,-0.039961025910546524,-0.0399611989543659,0.4210470792877,-0.03996078712641364,-0.03904622947022269,0.16398161252493304,-0.039964852736615104,-0.039959559173377956,-0.039944466833663726,-0.03995942506382456,0.16399883452304598,0.16398106083813027,0.16398247021767245,0.1639870949432965,0.16398132204928384,0.10372866962247493,-0.03996131849684828,0.22162650704413578,-0.03995157395059486,0.16398128647022694,0.3916794073500004,0.39399369872052276,-0.03995908835767528,-0.03987156180138553,-0.03996051668335943,-0.039960850662439124,-0.039962270532007786,-0.03995499953216428,0.4816014862944866,0.4008367376910227,0.19858231672678522,-0.03996165726299979,-0.039613730011062664,0.4003881702227942,-0.0399610384630232,0.16398180936835932,-0.03309394804059414,0.10372844788187331,0.39602972687903565,0.16398405687138276,0.10372896828020316,-0.03994815748926572,-0.03994891652445711,0.3673871180173061,0.4202979311173262,0.10372859473480613,0.16398107049719776,-0.03996353610424745,0.484289367324648,-0.039960806902665555,0.1637135045590994,0.10373113884664935,0.3712760506866068,-0.039948520046112754,0.40710525606079384,-0.03996010706908917,0.10372945815008063,-0.03996103842687931,0.10373052091010217,0.23620939808433514,0.39384525529990744,0.16398150696365935,0.4008453743550798,-0.03995621802463473,0.16398132647136018,-0.03995975440510015,0.4060216055587155,0.3673864520734285,0.3298699040230582,-0.039231601887798465,0.10369476077487401,0.10374048016889567,0.39384666888198355,-0.03996090586965426,0.3938462420776016,0.39602658995557866,0.11114651604972635,0.4167351715683532,-0.03996104996637599,0.10380078592357865,0.32986949221002526,-0.03996221038895554,0.39602596791581013,0.4366156488290989,-0.03966801077372963,-0.011845298008100694,-0.03996013507179229,0.16405384777720583,-0.039956840549150566,0.4167241732925587,0.3921217237928154,0.42317714802653233,0.3926478853241358,0.3938378232581642,0.10373319635302157,-0.03996116201792034,-0.03995588028119405,0.10373099816417737,0.10372995781391252,0.39384799075498783,0.3692598548095142,-0.03914841724543775,0.10887344362677827,0.10380027781992338,0.1639813303736001,-0.039177329035103604,-0.05234126262294955,-0.039960378316344566,-0.03995627210977342,-0.03995639310487196,-0.03995648571809792,0.39385104642227164,0.3309851411744999,0.10362914132259715,0.27289659105177544,0.32986890076800046,0.37128302985528994,-0.03996222307967478,0.10373973032844853,0.2728977433524043,0.3674065601815347,0.2728968050780941,0.40084993462580626,0.3985205339974891,0.39384572953198216,0.36808809344432797,0.10376287434495778,0.416758716481258,-0.03993100452209661,0.04877702923709634,0.10372988594555907,0.27289582449121963,0.2729063036959177,-0.039960191480320116,0.10372975695514731,0.16398108022731805,0.09812171515179205,0.46388710686005613,0.36165971291581517,0.3938586712135578,-0.040080318254528846,0.37127247802247837,0.22165103971050898,-0.0399591228532721,-0.03984159367358821,0.10374780303590983,0.3678505034965284,0.3939128984329846,-0.03995981594772559,0.2976042470233448,0.32986787015037444,0.40604068919962744,0.10373348528495656,0.3938451728949671,-0.011848279778699391,0.39602752623988474,0.2835313357086036,0.4060208423318277,0.3938899641875552,0.36837546735122806,0.4040044591894425,0.5268598062729638,0.39387134115731764,0.21090310793272732,0.34894658401844336,0.10372990937690578,-0.039959503727945406,0.4168515875834281,0.3329937176893189,0.24151038148607445,0.37351677924481036,0.16395627047382,0.10373020596785924,0.21090878345296066,0.33006946603691667,0.40602575297709054,0.16398133063980588,0.39428666295281417,0.40028415510222987,0.4002769566067849,0.3254947125523173,-0.03995512863219085,0.32988534899485505,0.329868950578392,0.19858921300620644,0.16398157075825137,-0.049745008623454115,0.16398653564519192,-0.03264343276027155,0.27286239868475487,0.40079691934635087,-0.039953932410305516,0.16398246575640113,-0.03879003235532032,0.132808249016216,-0.03996109526685421,-0.03995347883023429,0.400842295079162,0.2722614787846396,0.367386472562127,0.3938648430367473,0.32986975832513366],"type":"scatter3d"},{"customdata":[["Proof-of-BibleHash"],["SHA-256 + Hive"],["Proof-of-Authority"],["ECC 256K1"],["Leased POS"]],"hovertemplate":"<b>%{hovertext}</b><br><br>class=%{marker.color}<br>PC 1=%{x}<br>PC 2=%{y}<br>PC 3=%{z}<br>Algorithm=%{customdata[0]}<extra></extra>","hovertext":["BiblePay","LitecoinCash","Poa Network","Acute Angle Cloud","Waves"],"legendgroup":"1","marker":{"color":[1,1,1,1,1],"coloraxis":"coloraxis","symbol":"square"},"mode":"markers","name":"1","scene":"scene","showlegend":true,"x":[-0.2067518993123665,-0.31303292267008204,-0.40552347690083806,-0.42871919287788757,-0.4412079273073658],"y":[2.7063062569858873,2.6920044167257515,3.3218559331598057,3.63440711908042,3.145248709784864],"z":[9.314347905444528,16.397872124841516,19.025249517756066,9.11104602366278,12.717970330538257],"type":"scatter3d"},{"customdata":[["TRC10"]],"hovertemplate":"<b>%{hovertext}</b><br><br>class=%{marker.color}<br>PC 1=%{x}<br>PC 2=%{y}<br>PC 3=%{z}<br>Algorithm=%{customdata[0]}<extra></extra>","hovertext":["BitTorrent"],"legendgroup":"2","marker":{"color":[2],"coloraxis":"coloraxis","symbol":"x"},"mode":"markers","name":"2","scene":"scene","showlegend":true,"x":[34.07426714960088],"y":[1.7871427817022265],"z":[-0.49631056525927675],"type":"scatter3d"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"scene":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"xaxis":{"title":{"text":"PC 1"}},"yaxis":{"title":{"text":"PC 2"}},"zaxis":{"title":{"text":"PC 3"}}},"coloraxis":{"colorbar":{"title":{"text":"class"}},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"legend":{"title":{"text":"class"},"tracegroupgap":0,"x":0,"y":1},"margin":{"t":60},"width":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('58a5fc45-af4a-4771-a039-dfd908067cf2');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
# Create a table with tradable cryptocurrencies.
clustered_df.hvplot.table(sortable=True, selectable=True)
```






<div id='1002'>





  <div class="bk-root" id="bc04934a-ff98-43e5-bd60-329a1203613a" data-root-id="1002"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"e7b07c01-b6f8-4be1-8251-61b8f107d8aa":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01624","sizing_mode":"stretch_width"},"id":"1003","type":"Spacer"},{"attributes":{},"id":"1042","type":"IntEditor"},{"attributes":{},"id":"1054","type":"UnionRenderers"},{"attributes":{"format":"0,0.0[00000]"},"id":"1036","type":"NumberFormatter"},{"attributes":{"source":{"id":"1004"}},"id":"1053","type":"CDSView"},{"attributes":{},"id":"1005","type":"Selection"},{"attributes":{},"id":"1041","type":"NumberFormatter"},{"attributes":{"data":{"Algorithm":["Scrypt","Scrypt","X13","SHA-256","Ethash","Scrypt","X11","CryptoNight-V7","Ethash","Equihash","SHA-512","Multiple","SHA-256","SHA-256","Scrypt","X15","X11","Scrypt","Scrypt","Scrypt","Multiple","Scrypt","SHA-256","Scrypt","Scrypt","Scrypt","Quark","Groestl","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","X11","Scrypt","Groestl","Multiple","SHA-256","Scrypt","Scrypt","Scrypt","Scrypt","PoS","Scrypt","Scrypt","NeoScrypt","Scrypt","Scrypt","Scrypt","Scrypt","X11","Scrypt","X11","SHA-256","Scrypt","Scrypt","Scrypt","SHA3","Scrypt","HybridScryptHash256","Scrypt","Scrypt","SHA-256","Scrypt","X13","Scrypt","SHA-256","Scrypt","X13","NeoScrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","X11","X11","SHA-256","Multiple","SHA-256","PHI1612","X11","SHA-256","SHA-256","SHA-256","X11","Scrypt","Scrypt","Scrypt","Scrypt","Lyra2REv2","Scrypt","X11","Multiple","SHA-256","X13","Scrypt","CryptoNight","CryptoNight","Shabal256","Counterparty","Scrypt","SHA-256","Groestl","Scrypt","Scrypt","Scrypt","X13","Scrypt","Scrypt","Scrypt","Scrypt","X13","Scrypt","Stanford Folding","X11","Multiple","QuBit","Scrypt","Scrypt","Scrypt","M7 POW","Scrypt","SHA-256","Scrypt","X11","SHA3","X11","Lyra2RE","SHA-256","QUAIT","X11","X11","Scrypt","Scrypt","Scrypt","Ethash","X13","Blake2b","SHA-256","X15","X11","SHA-256","BLAKE256","Scrypt","1GB AES Pattern Search","SHA-256","X11","Scrypt","SHA-256","SHA-256","NIST5","Scrypt","Scrypt","X11","Dagger","Scrypt","X11GOST","X11","Scrypt","SHA-256","Scrypt","PoS","Scrypt","X11","X11","SHA-256","SHA-256","NIST5","X11","Scrypt","POS 3.0","Scrypt","Scrypt","Scrypt","X13","X11","X11","Equihash","X11","Scrypt","CryptoNight","SHA-256","SHA-256","X11","Scrypt","Multiple","Scrypt","Scrypt","Scrypt","SHA-256","Scrypt","Scrypt","SHA-256D","PoS","Scrypt","X11","Lyra2Z","PoS","X13","X14","PoS","SHA-256D","Ethash","Equihash","DPoS","X11","Scrypt","X11","X13","X11","PoS","Scrypt","Scrypt","X11","PoS","X11","SHA-256","Scrypt","X11","Scrypt","Scrypt","X11","CryptoNight","Scrypt","Scrypt","Scrypt","Scrypt","Quark","QuBit","Scrypt","CryptoNight","Lyra2RE","Scrypt","SHA-256","X11","Scrypt","X11","Scrypt","CryptoNight-V7","Scrypt","Scrypt","Scrypt","X13","X11","Equihash","Scrypt","Scrypt","Lyra2RE","Scrypt","Dagger-Hashimoto","X11","Blake2S","X11","Scrypt","PoS","X11","NIST5","PoS","X11","Scrypt","Scrypt","Scrypt","SHA-256","X11","Scrypt","Scrypt","SHA-256","PoS","Scrypt","X15","SHA-256","Scrypt","POS 3.0","CryptoNight-V7","536","Argon2d","Blake2b","Cloverhash","CryptoNight","NIST5","X11","NIST5","Skein","Scrypt","X13","Scrypt","X11","X11","Scrypt","CryptoNight","X13","Time Travel","Scrypt","Keccak","SkunkHash v2 Raptor","X11","Skein","SHA-256","X11","Scrypt","VeChainThor Authority","Scrypt","PoS","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","Scrypt","CryptoNight","SHA-512","Ouroboros","X11","Equihash","NeoScrypt","X11","Scrypt","NeoScrypt","Lyra2REv2","Equihash","Scrypt","SHA-256","NIST5","PHI1612","Dagger","Scrypt","Quark","Scrypt","POS 2.0","Scrypt","SHA-256","X11","NeoScrypt","Ethash","NeoScrypt","X11","DPoS","NIST5","X13","Multiple","Scrypt","CryptoNight","CryptoNight","Ethash","NIST5","Quark","X11","CryptoNight-V7","Scrypt","Scrypt","Scrypt","X11","BLAKE256","X11","NeoScrypt","Quark","NeoScrypt","Scrypt","Scrypt","Scrypt","X11","X11","SHA-256","C11","POS 3.0","Ethash","Scrypt","CryptoNight","SkunkHash","Scrypt","CryptoNight","Scrypt","Dagger","Lyra2REv2","X13","Proof-of-BibleHash","SHA-256 + Hive","Scrypt","Scrypt","X11","C11","Proof-of-Authority","X11","XEVAN","Scrypt","VBFT","Ethash","CryptoNight","Scrypt","IMesh","NIST5","Scrypt","Scrypt","Equihash","Scrypt","Lyra2Z","Green Protocol","PoS","Scrypt","Semux BFT consensus","X11","Quark","PoS","CryptoNight","X16R","Scrypt","NIST5","Lyra2RE","XEVAN","Tribus","Scrypt","Lyra2Z","CryptoNight","CryptoNight Heavy","CryptoNight","Scrypt","Scrypt","Jump Consistent Hash","SHA-256D","CryptoNight","Scrypt","X15","Scrypt","Quark","SHA-256","DPoS","X16R","HMQ1725","X11","X16R","Quark","Quark","Scrypt","Lyra2REv2","Quark","Scrypt","Scrypt","CryptoNight-V7","Cryptonight-GPU","XEVAN","CryptoNight Heavy","X11","X11","Scrypt","PoS","SHA-256","Keccak","X11","X11","Scrypt","SHA-512","X16R","ECC 256K1","Equihash","XEVAN","Lyra2Z","SHA-256","XEVAN","X11","CryptoNight","Quark","Blake","Blake","Equihash","Exosis","Scrypt","Scrypt","Equihash","Quark","Equihash","Quark","Scrypt","QuBit","X11","Scrypt","XEVAN","SHA-256D","X11","SHA-256","X13","SHA-256","X11","DPoS","Scrypt","Scrypt","X11","NeoScrypt","Scrypt","Blake","Scrypt","SHA-256","Scrypt","X11","Scrypt","Scrypt","SHA-256","X11","SHA-256","Scrypt","Scrypt","Scrypt","Groestl","X11","Scrypt","PoS","Scrypt","Scrypt","X11","SHA-256","DPoS","Scrypt","Scrypt","NeoScrypt","Multiple","X13","Equihash+Scrypt","DPoS","Ethash","DPoS","SHA-256","Leased POS","PoS","TRC10","PoS","SHA-256","Scrypt","CryptoNight","Equihash","Scrypt"],"CoinName":["42 Coin","404Coin","EliteCoin","Bitcoin","Ethereum","Litecoin","Dash","Monero","Ethereum Classic","ZCash","Bitshares","DigiByte","BitcoinDark","PayCoin","ProsperCoin","KoboCoin","Spreadcoin","Argentum","Aurora Coin","BlueCoin","MyriadCoin","MoonCoin","ZetaCoin","SexCoin","Quatloo","EnergyCoin","QuarkCoin","Riecoin","Digitalcoin ","BitBar","Catcoin","CryptoBullion","CannaCoin","CryptCoin","CasinoCoin","Diamond","Verge","DevCoin","EarthCoin","E-Gulden","Einsteinium","Emerald","Exclusive Coin","FlutterCoin","Franko","FeatherCoin","GrandCoin","GlobalCoin","GoldCoin","HoboNickels","HyperStake","Infinite Coin","IOCoin","IXcoin","KrugerCoin","LuckyCoin","Litebar ","MaxCoin","MegaCoin","MediterraneanCoin","MintCoin","MinCoin","MazaCoin","Nautilus Coin","NavCoin","NobleCoin","Namecoin","NyanCoin","OpalCoin","Orbitcoin","PotCoin","PhoenixCoin","Reddcoin","RonPaulCoin","StableCoin","SmartCoin","SuperCoin","SyncCoin","SysCoin","TeslaCoin","TigerCoin","TittieCoin","TorCoin","TerraCoin","UnbreakableCoin","Unobtanium","UroCoin","UnitaryStatus Dollar","UltraCoin","ViaCoin","VeriCoin","Vertcoin","WorldCoin","X11 Coin","Crypti","JouleCoin","StealthCoin","ZCC Coin","ByteCoin","DigitalNote ","BurstCoin","StorjCoin","MonaCoin","Neutron","FairCoin","Gulden","RubyCoin","PesetaCoin","Kore","Wild Beast Coin","Dnotes","Flo","8BIT Coin","Sativa Coin","ArtByte","Folding Coin","Ucoin","Unitus","CypherPunkCoin","OmniCron","Vtorrent","GreenCoin","Cryptonite","MasterCoin","SoonCoin","1Credit","IslaCoin","Nexus","MarsCoin ","Crypto","Anarchists Prime","Droidz","BowsCoin","Squall Coin","Song Coin","BitZeny","Diggits","Expanse","Paycon","Siacoin","Emercoin","EverGreenCoin","MindCoin","I0coin","Decred","Revolution VR","HOdlcoin","EDRCoin","Hitcoin","Gamecredits","DubaiCoin","CarpeDiemCoin","PWR Coin","BillaryCoin","GPU Coin","Adzcoin","SoilCoin","YoCoin","SibCoin","EuropeCoin","ZeitCoin","SwingCoin","SafeExchangeCoin","Nebuchadnezzar","Francs","BolivarCoin","Ratecoin","Revenu","Clockcoin","VIP Tokens","BitSend","Omni","Let it Ride","PutinCoin","iBankCoin","Frankywillcoin","MudraCoin","PizzaCoin","Lutetium Coin","Komodo","GoldBlocks","CarterCoin","Karbo","BitTokens","ZayedCoin","MustangCoin","ZoneCoin","Circuits of Value","RootCoin","DopeCoin","BitCurrency","DollarCoin","Swiscoin","Shilling","BuzzCoin","Opair","PesoBit","Halloween Coin","ZCoin","CoffeeCoin","RoyalCoin","GanjaCoin V2","TeamUP","LanaCoin","Elementrem","ZClassic","ARK","InsaneCoin","KiloCoin","ArtexCoin","EmberCoin","XenixCoin","FreeCoin","PLNCoin","AquariusCoin","Kurrent","Creatio","Eternity","Eurocoin","BitcoinFast","Stakenet","BitConnect Coin","MoneyCoin","Enigma","Cannabis Industry Coin","Russiacoin","PandaCoin","GameUnits","GAKHcoin","Allsafe","LiteCreed","OsmiumCoin","Bikercoins","HexxCoin","Klingon Empire Darsek","Internet of People","KushCoin","Printerium","PacCoin","Impeach","Citadel","Zilbercoin","FirstCoin","BeaverCoin","FindCoin","VaultCoin","Zero","OpenChat","Canada eCoin","Zoin","RenosCoin","DubaiCoin","VirtacoinPlus","TajCoin","Impact","EB3coin","Atmos","HappyCoin","Coinonat","MacronCoin","Condensate","Independent Money System","ArgusCoin","LomoCoin","ProCurrency","GoldReserve","BenjiRolls","GrowthCoin","ILCoin","Phreak","Degas Coin","HTML5 Coin","Ultimate Secure Cash","EquiTrader","QTUM","Quantum Resistant Ledger","Espers","Dynamic","Nano","ChanCoin","Dinastycoin","Denarius","DigitalPrice","Virta Unique Coin","Bitcoin Planet","Unify","BritCoin","SocialCoin","ArcticCoin","DAS","Linda","LeviarCoin","DeepOnion","Bitcore","gCn Coin","SmartCash","Signatum","Onix","Cream","Bitcoin Cash","Monoeci","Draftcoin","Vechain","Sojourn Coin","Stakecoin","NewYorkCoin","FrazCoin","Kronecoin","AdCoin","Linx","CoinonatX","Ethereum Dark","Sumokoin","Obsidian","Cardano","Regalcoin","BitcoinZ","TrezarCoin","Elements","TerraNovaCoin","VIVO Coin","Rupee","Bitcoin Gold","WomenCoin","Theresa May Coin","NamoCoin","LUXCoin","Pirl","Xios","Bitcloud 2.0","eBoost","KekCoin","BlackholeCoin","Infinity Economics","Pura","Innova","Ellaism","GoByte","Magnet","Lamden Tau","Electra","Bitcoin Diamond","SHIELD","Cash & Back Coin","UltraNote","BitCoal","DaxxCoin","Bulwark","Kalkulus","AC3","Lethean","GermanCoin","LiteCoin Ultra","PopularCoin","PhantomX","Photon","Sucre","SparksPay","Digiwage","GunCoin","IrishCoin","Trollcoin","Litecoin Plus","Monkey Project","Pioneer Coin","UnitedBitcoin","Interzone","TokenPay","1717 Masonic Commemorative Token","My Big Coin","TurtleCoin","MUNcoin","Unified Society USDEX","Niobio Cash","ShareChain","Travelflex","KREDS","Tokyo Coin","BiblePay","LitecoinCash","BitFlip","LottoCoin","Crypto Improvement Fund","Stipend","Poa Network","Pushi","Ellerium","Velox","Ontology","Callisto Network","BitTube","Poseidon","Aidos Kuneen","Bitspace","Briacoin","Ignition","Bitrolium","MedicCoin","Alpenschillling","Bitcoin Green","Deviant Coin","Abjcoin","Semux","FuturoCoin","Carebit","Zealium","Monero Classic","Proton","iDealCash","Jumpcoin","Infinex","Bitcoin Incognito","KEYCO","HollyWoodCoin","GINcoin","PlatinCoin","Loki","Newton Coin","Swisscoin","Xt3ch","MassGrid","TheVig","PluraCoin","EmaratCoin","Dekado","Lynx","Poseidon Quark","BitcoinWSpectrum","Muse","Motion","PlusOneCoin","Axe","Trivechain","Dystem","Giant","Peony Coin","Absolute Coin","Vitae","HexCoin","TPCash","Webchain","Ryo","Urals Coin","Qwertycoin","ARENON","EUNO","MMOCoin","Ketan","Project Pai","XDNA","PAXEX","Azart","ThunderStake","Kcash","Xchange","Acute Angle Cloud","CrypticCoin","Bettex coin","Actinium","Bitcoin SV","BitMoney","Junson Ming Chan Coin","FREDEnergy","HerbCoin","Universal Molecule","Lithium","PirateCash","Exosis","Block-Logic","Oduwa","Beam","Galilel","Bithereum","Crypto Sports","Credit","SLICE","Dash Platinum","Nasdacoin","Beetle Coin","Titan Coin","Award","BLAST","Bitcoin Rhodium","GlobalToken","Insane Coin","ALAX","LiteDoge","SolarCoin","TruckCoin","UFO Coin","OrangeCoin","BlakeCoin","BitstarCoin","NeosCoin","HyperCoin","PinkCoin","Crypto Escudo","AudioCoin","IncaKoin","Piggy Coin","Crown Coin","Genstake","SmileyCoin","XiaoMiCoin","Groestlcoin","CapriCoin"," ClubCoin","Radium","Bata","Pakcoin","Creditbit ","OKCash","Lisk","HiCoin","WhiteCoin","FriendshipCoin","JoinCoin","Triangles Coin","Vollar","EOS","Reality Clash","Oxycoin","TigerCash","Waves","Particl","BitTorrent","Nxt","ZEPHYR","Gapcoin","Beldex","Horizen","BitcoinPlus"],"PC_1":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[532]},"PC_2":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[532]},"PC_3":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[532]},"ProofType":["PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoC","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoW","PoS","PoS/PoW/PoT","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoS","PoW","PoW/PoS","PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoS","PoW/PoS","PoW","PoS","PoW","PoS","PoW/PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoST","PoW","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoC","PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW/nPoS","PoW","PoW","PoW","PoW/PoS","PoW","PoS/PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoS","PoW/PoS","PoC","PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoS","PoW","PoS","dPoW/PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoW","PoW","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoS","PoW/PoS","PoS","PoW/PoS","PoW","PoW","DPoS","PoW/PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","TPoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW/PoS ","PoW","PoS","PoW","PoW","PoW/PoS","PoW","PoW","PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoS","PoW/PoS","PoW","PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoS","PoW","PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoS","Proof of Authority","PoW","PoS","PoW","PoW","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoS","PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoS","PoW","PoW","PoW","PoW","PoW/PoS","DPoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoS","PoW","PoW","PoW/PoS","PoW/PoS","PoS","PoW","PoW","PoW","PoS","PoS","PoW and PoS","PoW","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoS","POBh","PoW + Hive","PoW","PoW","PoW","PoW/PoS","PoA","PoW/PoS","PoW/PoS","PoS","PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoS","PoW","PoS","PoS","PoW/PoS","DPoS","PoW","PoW/PoS","PoS","PoW","PoS","PoW/PoS","PoW","PoW","PoS/PoW","PoW","PoS","PoW","PoW","PoW","PoW","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoS","HPoW","PoS","PoS","PoS","PoW","PoW","PoW","PoW/PoS","PoS","PoW/PoS","PoS","PoW/PoS","PoS","PoW","PoW/PoS","PoW","PoW","PoW","PoW","PoS","PoW/PoS","PoS","PoS","PoW","PoW/PoS","PoS","PoW","PoW/PoS","Zero-Knowledge Proof","PoW","DPOS","PoW","PoS","PoW","PoW","Pos","PoS","PoW","PoW/PoS","PoW","PoW","PoS","PoW","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoS","PoW/PoS","PoW","PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW","PoW","PoW/PoS","DPoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW","PoW/PoS","PoW/PoS","PoS","PoW","PoW","Proof of Trust","PoW/PoS","DPoS","PoS","PoW/PoS","PoW/PoS","PoW","PoW/PoS","PoW","DPoS","PoW","DPoS","PoS","LPoS","PoS","DPoS","PoS/LPoS","DPoS","PoW/PoS","PoW","PoW","PoS"],"TotalCoinSupply":["42","532000000","314159265359","21000000","0","84000000","22000000","0","210000000","21000000","3600570502","21000000000","22000000","12500000","21000000","350000000","20000000","64000000","16768584","0","2000000000","384000000000","169795588","250000000","100000000","0","247000000","84000000","48166000","500000","21000000 ","1000000","13140000","18000000","40000000000","4380000","16555000000","21000000000","13500000000","21000000 ","299792458","32000000","0","0","11235813","336000000","1420609614","70000000","72245700","120000000","0","90600000000","22000000","21000000","265420800","20000000","1350000","100000000","42000000","200000000","0","10000000","2419200000","16180000","0","15000000000","21000000","337000000","0","3770000","420000000","98000000","0","21000000","250000000","51200000","0","1000","888000000","100000000","47011968","2300000000","10000000","42000000","80000000","250000","0","1600000000","100000000","23000000","0","84000000","265420800","5500000","0","45000000","0","1000000000","184467440735","10000000000","2158812800","500000000","105120000","68000000","0","1680000000","0","166386000","12000000","2628000","500000000","160000000","0","10000000","1000000000","1000000000","20000000","0","0","3371337","20000000","10000000000","1840000000","619478","21000000","92000000000","0","78000000","33000000","65789100","53760000","5060000","21000000","0","210240000","250000000","100000000","16906397","50000000","0","1000000000","26298000","16000000","21000000","21000000","210000000","81962100","22000000","26550000000","84000000","10500000","21626280000 ","0","42000000","221052632","84000000","30000000","168351300","24000000","384000000"," 99000000000","40000000","2147483647","20000000","20000000","25000000","75000000","222725000","525000000","90000000","139000000","616448","33500000","2000000000","44333333","100000000","200000000","25000000","657000000","200000000","50000000","90000000","10000000","21000000","9736000","3000000","21000000","1200000000","0","200000000","0","10638298","3100000000","30000000","20000000000","74000000","0","1500000000","21400000","39999898","2500124","100000000","301000000","7506000000","26205539","21000000","125000000","30000000","10000000000","500000000","850000000","3853326.77707314","50000000","38540000 ","42000000","228000000","20000000","60000000","20000000","33000000","76500000","28000000","650659833","5000000","21000000","144000000","32514916898","13000000","3315789","15000000","78835200","2714286","25000000","9999999","500000000","21000000","9354000","20000000","100000000000","21933333","185000000","55000000","110000000","3360000","14524851.4827","1000000000","17000000","1000000000","100000000 ","21000000","34426423","2232901","100000000","36900000","110000000","4000000000","110290030","100000000","48252000","400000000","500000000","21212121","28600000","1000000000","75000000000","40000000","35520400","2000000000","2500000000","30000000","105000000","90000000000","200084200","72000000","100000000","105000000","50000000000","0","340282367","30000000","2000000000","10000000","100000000","120000000","100000000","19276800","30000000"," 75000000","60000000","18900000","50000000000","54000000","18898187.6216583","21000000","200000000000","5000000000","137500000","1100000000","100000000","21000000","9507271","17405891.19707116","86712634466","10500000000","61599965","0","20000000","84000000","100000000","100000000","48252000","4200000","88888888","91388946","45000000000","27000000","21000000000","400000000","1800000000","15733333","27000000","24000000","21000000","25000000000","100000000","1200000000","60000000","156306732.71","21000000","200000000","100000000","21000000","14788275.991","9000000000","350000000","45000000","280000000","31800000","144000000","500000000","30000000000","210000000","660000000","210000000","85000000000","12500000","10000000000","27716121","20000000","550000000","999481516","50000000000","150000000","4999999999","50000000"," 90000000000","19800000","21000000","120000000","500000000","64000000","900000000","4000000","21000000","23000000","20166000","23000000","25000000","1618033","30000000","1000000000000","16600000","232000000","336000000","10000000000","100000000","1100000000","800000000","5200000000","840000000","40000000","18406979840","500000000","19340594","252460800","25000000","60000000","124000000","1000000000","6500000000","1000000000","21000000","25000000","50000000","3000000","5000000","70000000","500000000","300000000","21000000","88000000","30000000","100000000","100000000","200000000","80000000","18400000","45000000","5121951220","21000000","26280000","21000000","18000000","26000000","10500000","600000518","150000000","184000000000","10200000000","44000000","168000000","100000000","1000000000","84000000","90000000","92000000000","650000000 ","100262205","18081806 ","22075700","21000000","21000000","82546564","21000000","5151000","16880000000","52500000","100000000","22105263","1000000000","1750000000","88188888","210000000","184470000000","55000000","50000000","260000000","210000000","2100000000","366000000","100000000","25000000","18000000000","1000000000","100000000","1000000000","7600000000","50000000","84000000","21000000","70000000000","0","8080000000","54000000","105120001.44","25228800","105000000","21000000","120000000","21000000","262800000","19035999","30886000","13370000","74800000000","100000000","19700000","84000000","500000000","5000000000","420000000","64000000","2100000","168000000","30000000","1000000000","35000000000","98100000000","0","4000000000","200000000","7000000000","54256119","21000000","0","500000000","1000000000","10500000000","190000000","1000000000","42000000","15000000","50000000000","400000000","105000000","208000000","160000000","9000000","5000000","182000000","16504333","105000000","159918400","10008835635","300000000","60168145","2800000","120000","2100000000","0","24487944","0","1000000000","100000000","8634140","990000000000","1000000000","2000000000","250000000","1400222610","21000000","1000000"],"TotalCoinsMined":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[532]},"class":{"__ndarray__":"AAAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAAAAAAMAAAADAAAAAwAAAAAAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAMAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAwAAAAMAAAADAAAAAAAAAAMAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAAAAAADAAAAAAAAAAMAAAADAAAAAwAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAMAAAADAAAAAwAAAAMAAAAAAAAAAQAAAAEAAAADAAAAAwAAAAMAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAABAAAAAwAAAAAAAAADAAAAAwAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAAAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAMAAAADAAAAAAAAAA==","dtype":"int32","order":"little","shape":[532]}},"selected":{"id":"1005"},"selection_policy":{"id":"1054"}},"id":"1004","type":"ColumnDataSource"},{"attributes":{"editor":{"id":"1042"},"field":"class","formatter":{"id":"1041"},"title":"class"},"id":"1043","type":"TableColumn"},{"attributes":{},"id":"1047","type":"StringEditor"},{"attributes":{"editor":{"id":"1047"},"field":"CoinName","formatter":{"id":"1046"},"title":"CoinName"},"id":"1048","type":"TableColumn"},{"attributes":{"children":[{"id":"1003"},{"id":"1051"},{"id":"1058"}],"margin":[0,0,0,0],"name":"Row01620","tags":["embedded"]},"id":"1002","type":"Row"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01625","sizing_mode":"stretch_width"},"id":"1058","type":"Spacer"},{"attributes":{"columns":[{"id":"1008"},{"id":"1013"},{"id":"1018"},{"id":"1023"},{"id":"1028"},{"id":"1033"},{"id":"1038"},{"id":"1043"},{"id":"1048"}],"height":300,"reorderable":false,"source":{"id":"1004"},"view":{"id":"1053"},"width":700},"id":"1051","type":"DataTable"},{"attributes":{},"id":"1011","type":"StringFormatter"},{"attributes":{},"id":"1021","type":"StringFormatter"},{"attributes":{},"id":"1046","type":"StringFormatter"},{"attributes":{"editor":{"id":"1007"},"field":"Algorithm","formatter":{"id":"1006"},"title":"Algorithm"},"id":"1008","type":"TableColumn"},{"attributes":{},"id":"1007","type":"StringEditor"},{"attributes":{},"id":"1006","type":"StringFormatter"},{"attributes":{},"id":"1012","type":"StringEditor"},{"attributes":{},"id":"1017","type":"NumberEditor"},{"attributes":{"editor":{"id":"1017"},"field":"TotalCoinsMined","formatter":{"id":"1016"},"title":"TotalCoinsMined"},"id":"1018","type":"TableColumn"},{"attributes":{"format":"0,0.0[00000]"},"id":"1026","type":"NumberFormatter"},{"attributes":{"format":"0,0.0[00000]"},"id":"1016","type":"NumberFormatter"},{"attributes":{},"id":"1027","type":"NumberEditor"},{"attributes":{"editor":{"id":"1012"},"field":"ProofType","formatter":{"id":"1011"},"title":"ProofType"},"id":"1013","type":"TableColumn"},{"attributes":{"editor":{"id":"1032"},"field":"PC_2","formatter":{"id":"1031"},"title":"PC 2"},"id":"1033","type":"TableColumn"},{"attributes":{},"id":"1022","type":"StringEditor"},{"attributes":{"editor":{"id":"1027"},"field":"PC_1","formatter":{"id":"1026"},"title":"PC 1"},"id":"1028","type":"TableColumn"},{"attributes":{"editor":{"id":"1022"},"field":"TotalCoinSupply","formatter":{"id":"1021"},"title":"TotalCoinSupply"},"id":"1023","type":"TableColumn"},{"attributes":{},"id":"1032","type":"NumberEditor"},{"attributes":{},"id":"1037","type":"NumberEditor"},{"attributes":{"format":"0,0.0[00000]"},"id":"1031","type":"NumberFormatter"},{"attributes":{"editor":{"id":"1037"},"field":"PC_3","formatter":{"id":"1036"},"title":"PC 3"},"id":"1038","type":"TableColumn"}],"root_ids":["1002"]},"title":"Bokeh Application","version":"2.4.2"}};
    var render_items = [{"docid":"e7b07c01-b6f8-4be1-8251-61b8f107d8aa","root_ids":["1002"],"roots":{"1002":"bc04934a-ff98-43e5-bd60-329a1203613a"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>




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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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






<div id='1070'>





  <div class="bk-root" id="6d07fec8-db25-4ace-9a23-01d7c5335c07" data-root-id="1070"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"620cbc25-7be6-4956-ac3c-2ebc74c57fb4":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"GridStack1","overrides":[],"properties":[{"default":"warn","kind":null,"name":"mode"},{"default":null,"kind":null,"name":"ncols"},{"default":null,"kind":null,"name":"nrows"},{"default":true,"kind":null,"name":"allow_resize"},{"default":true,"kind":null,"name":"allow_drag"},{"default":[],"kind":null,"name":"state"}]},{"extends":null,"module":null,"name":"click1","overrides":[],"properties":[{"default":"","kind":null,"name":"terminal_output"},{"default":"","kind":null,"name":"debug_name"},{"default":0,"kind":null,"name":"clears"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"coordinates":null,"group":null,"text_color":"black","text_font_size":"12pt"},"id":"1079","type":"Title"},{"attributes":{},"id":"1202","type":"UnionRenderers"},{"attributes":{},"id":"1136","type":"Selection"},{"attributes":{},"id":"1083","type":"LinearScale"},{"attributes":{},"id":"1085","type":"LinearScale"},{"attributes":{"axis":{"id":"1087"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"1090","type":"Grid"},{"attributes":{"axis_label":"TotalCoinsMined","coordinates":null,"formatter":{"id":"1109"},"group":null,"major_label_policy":{"id":"1110"},"ticker":{"id":"1088"}},"id":"1087","type":"LinearAxis"},{"attributes":{},"id":"1110","type":"AllLabels"},{"attributes":{},"id":"1176","type":"UnionRenderers"},{"attributes":{"source":{"id":"1157"}},"id":"1164","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.2},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1162","type":"Scatter"},{"attributes":{},"id":"1088","type":"BasicTicker"},{"attributes":{"coordinates":null,"data_source":{"id":"1157"},"glyph":{"id":"1160"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1162"},"nonselection_glyph":{"id":"1161"},"selection_glyph":{"id":"1180"},"view":{"id":"1164"}},"id":"1163","type":"GlyphRenderer"},{"attributes":{"axis_label":"TotalCoinSupply","coordinates":null,"formatter":{"id":"1112"},"group":null,"major_label_policy":{"id":"1113"},"ticker":{"id":"1092"}},"id":"1091","type":"LinearAxis"},{"attributes":{"axis":{"id":"1091"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"1094","type":"Grid"},{"attributes":{},"id":"1092","type":"BasicTicker"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1100","type":"BoxAnnotation"},{"attributes":{},"id":"1095","type":"SaveTool"},{"attributes":{},"id":"1096","type":"PanTool"},{"attributes":{"data":{"CoinName":["Bitcoin","Ethereum","Litecoin","Monero","Ethereum Classic","ZCash","DigiByte","ProsperCoin","Spreadcoin","Argentum","MyriadCoin","MoonCoin","ZetaCoin","SexCoin","Quatloo","QuarkCoin","Riecoin","Digitalcoin ","Catcoin","CannaCoin","CryptCoin","Verge","DevCoin","EarthCoin","E-Gulden","Einsteinium","Emerald","Franko","FeatherCoin","GrandCoin","GlobalCoin","GoldCoin","Infinite Coin","IXcoin","KrugerCoin","LuckyCoin","Litebar ","MegaCoin","MediterraneanCoin","MinCoin","NobleCoin","Namecoin","NyanCoin","RonPaulCoin","StableCoin","SmartCoin","SysCoin","TigerCoin","TerraCoin","UnbreakableCoin","Unobtanium","UroCoin","ViaCoin","Vertcoin","WorldCoin","JouleCoin","ByteCoin","DigitalNote ","MonaCoin","Gulden","PesetaCoin","Wild Beast Coin","Flo","ArtByte","Folding Coin","Unitus","CypherPunkCoin","OmniCron","GreenCoin","Cryptonite","MasterCoin","SoonCoin","1Credit","MarsCoin ","Crypto","Anarchists Prime","BowsCoin","Song Coin","BitZeny","Expanse","Siacoin","MindCoin","I0coin","Revolution VR","HOdlcoin","Gamecredits","CarpeDiemCoin","Adzcoin","SoilCoin","YoCoin","SibCoin","Francs","BolivarCoin","Omni","PizzaCoin","Komodo","Karbo","ZayedCoin","Circuits of Value","DopeCoin","DollarCoin","Shilling","ZCoin","Elementrem","ZClassic","KiloCoin","ArtexCoin","Kurrent","Cannabis Industry Coin","OsmiumCoin","Bikercoins","HexxCoin","PacCoin","Citadel","BeaverCoin","VaultCoin","Zero","Canada eCoin","Zoin","DubaiCoin","EB3coin","Coinonat","BenjiRolls","ILCoin","EquiTrader","Quantum Resistant Ledger","Dynamic","Nano","ChanCoin","Dinastycoin","DigitalPrice","Unify","SocialCoin","ArcticCoin","DAS","LeviarCoin","Bitcore","gCn Coin","SmartCash","Onix","Bitcoin Cash","Sojourn Coin","NewYorkCoin","FrazCoin","Kronecoin","AdCoin","Linx","Sumokoin","BitcoinZ","Elements","VIVO Coin","Bitcoin Gold","Pirl","eBoost","Pura","Innova","Ellaism","GoByte","SHIELD","UltraNote","BitCoal","DaxxCoin","AC3","Lethean","PopularCoin","Photon","Sucre","SparksPay","GunCoin","IrishCoin","Pioneer Coin","UnitedBitcoin","Interzone","TurtleCoin","MUNcoin","Niobio Cash","ShareChain","Travelflex","KREDS","BitFlip","LottoCoin","Crypto Improvement Fund","Callisto Network","BitTube","Poseidon","Aidos Kuneen","Bitrolium","Alpenschillling","FuturoCoin","Monero Classic","Jumpcoin","Infinex","KEYCO","GINcoin","PlatinCoin","Loki","Newton Coin","MassGrid","PluraCoin","Motion","PlusOneCoin","Axe","HexCoin","Webchain","Ryo","Urals Coin","Qwertycoin","Project Pai","Azart","Xchange","CrypticCoin","Actinium","Bitcoin SV","FREDEnergy","Universal Molecule","Lithium","Exosis","Block-Logic","Beam","Bithereum","SLICE","BLAST","Bitcoin Rhodium","GlobalToken","SolarCoin","UFO Coin","BlakeCoin","Crypto Escudo","Crown Coin","SmileyCoin","Groestlcoin","Bata","Pakcoin","JoinCoin","Vollar","Reality Clash","Beldex","Horizen"],"TotalCoinSupply":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[238]},"TotalCoinsMined":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[238]},"class":[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]},"selected":{"id":"1182"},"selection_policy":{"id":"1202"}},"id":"1181","type":"ColumnDataSource"},{"attributes":{},"id":"1097","type":"WheelZoomTool"},{"attributes":{"label":{"value":"2"},"renderers":[{"id":"1163"}]},"id":"1179","type":"LegendItem"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.1},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1161","type":"Scatter"},{"attributes":{"overlay":{"id":"1100"}},"id":"1098","type":"BoxZoomTool"},{"attributes":{},"id":"1099","type":"ResetTool"},{"attributes":{"label":{"value":"1"},"renderers":[{"id":"1141"}]},"id":"1155","type":"LegendItem"},{"attributes":{},"id":"1129","type":"UnionRenderers"},{"attributes":{},"id":"1158","type":"Selection"},{"attributes":{"fill_color":{"value":"#fc4f30"},"hatch_color":{"value":"#fc4f30"},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1138","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.1},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1118","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.1},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1139","type":"Scatter"},{"attributes":{"coordinates":null,"data_source":{"id":"1135"},"glyph":{"id":"1138"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1140"},"nonselection_glyph":{"id":"1139"},"selection_glyph":{"id":"1156"},"view":{"id":"1142"}},"id":"1141","type":"GlyphRenderer"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"1133"},{"id":"1155"},{"id":"1179"},{"id":"1205"}],"location":[0,0],"title":"class"},"id":"1132","type":"Legend"},{"attributes":{},"id":"1152","type":"UnionRenderers"},{"attributes":{"data":{"CoinName":["BitTorrent"],"TotalCoinSupply":{"__ndarray__":"rkfhehSu7z8=","dtype":"float64","order":"little","shape":[1]},"TotalCoinsMined":{"__ndarray__":"AAAAAAAA8D8=","dtype":"float64","order":"little","shape":[1]},"class":[2]},"selected":{"id":"1158"},"selection_policy":{"id":"1176"}},"id":"1157","type":"ColumnDataSource"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#6d904f"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#6d904f"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1206","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.2},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1140","type":"Scatter"},{"attributes":{},"id":"1113","type":"AllLabels"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.2},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1119","type":"Scatter"},{"attributes":{"source":{"id":"1135"}},"id":"1142","type":"CDSView"},{"attributes":{"coordinates":null,"data_source":{"id":"1114"},"glyph":{"id":"1117"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1119"},"nonselection_glyph":{"id":"1118"},"selection_glyph":{"id":"1134"},"view":{"id":"1121"}},"id":"1120","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"1071"},{"id":"1078"},{"id":"1351"}],"margin":[0,0,0,0],"name":"Row01833","tags":["embedded"]},"id":"1070","type":"Row"},{"attributes":{"data":{"CoinName":["42 Coin","404Coin","EliteCoin","Dash","Bitshares","BitcoinDark","PayCoin","KoboCoin","Aurora Coin","BlueCoin","EnergyCoin","BitBar","CryptoBullion","CasinoCoin","Diamond","Exclusive Coin","FlutterCoin","HoboNickels","HyperStake","IOCoin","MaxCoin","MintCoin","MazaCoin","Nautilus Coin","NavCoin","OpalCoin","Orbitcoin","PotCoin","PhoenixCoin","Reddcoin","SuperCoin","SyncCoin","TeslaCoin","TittieCoin","TorCoin","UnitaryStatus Dollar","UltraCoin","VeriCoin","X11 Coin","Crypti","StealthCoin","ZCC Coin","BurstCoin","StorjCoin","Neutron","FairCoin","RubyCoin","Kore","Dnotes","8BIT Coin","Sativa Coin","Ucoin","Vtorrent","IslaCoin","Nexus","Droidz","Squall Coin","Diggits","Paycon","Emercoin","EverGreenCoin","Decred","EDRCoin","Hitcoin","DubaiCoin","PWR Coin","BillaryCoin","GPU Coin","EuropeCoin","ZeitCoin","SwingCoin","SafeExchangeCoin","Nebuchadnezzar","Ratecoin","Revenu","Clockcoin","VIP Tokens","BitSend","Let it Ride","PutinCoin","iBankCoin","Frankywillcoin","MudraCoin","Lutetium Coin","GoldBlocks","CarterCoin","BitTokens","MustangCoin","ZoneCoin","RootCoin","BitCurrency","Swiscoin","BuzzCoin","Opair","PesoBit","Halloween Coin","CoffeeCoin","RoyalCoin","GanjaCoin V2","TeamUP","LanaCoin","ARK","InsaneCoin","EmberCoin","XenixCoin","FreeCoin","PLNCoin","AquariusCoin","Creatio","Eternity","Eurocoin","BitcoinFast","Stakenet","BitConnect Coin","MoneyCoin","Enigma","Russiacoin","PandaCoin","GameUnits","GAKHcoin","Allsafe","LiteCreed","Klingon Empire Darsek","Internet of People","KushCoin","Printerium","Impeach","Zilbercoin","FirstCoin","FindCoin","OpenChat","RenosCoin","VirtacoinPlus","TajCoin","Impact","Atmos","HappyCoin","MacronCoin","Condensate","Independent Money System","ArgusCoin","LomoCoin","ProCurrency","GoldReserve","GrowthCoin","Phreak","Degas Coin","HTML5 Coin","Ultimate Secure Cash","QTUM","Espers","Denarius","Virta Unique Coin","Bitcoin Planet","BritCoin","Linda","DeepOnion","Signatum","Cream","Monoeci","Draftcoin","Vechain","Stakecoin","CoinonatX","Ethereum Dark","Obsidian","Cardano","Regalcoin","TrezarCoin","TerraNovaCoin","Rupee","WomenCoin","Theresa May Coin","NamoCoin","LUXCoin","Xios","Bitcloud 2.0","KekCoin","BlackholeCoin","Infinity Economics","Magnet","Lamden Tau","Electra","Bitcoin Diamond","Cash & Back Coin","Bulwark","Kalkulus","GermanCoin","LiteCoin Ultra","PhantomX","Digiwage","Trollcoin","Litecoin Plus","Monkey Project","TokenPay","1717 Masonic Commemorative Token","My Big Coin","Unified Society USDEX","Tokyo Coin","Stipend","Pushi","Ellerium","Velox","Ontology","Bitspace","Briacoin","Ignition","MedicCoin","Bitcoin Green","Deviant Coin","Abjcoin","Semux","Carebit","Zealium","Proton","iDealCash","Bitcoin Incognito","HollyWoodCoin","Swisscoin","Xt3ch","TheVig","EmaratCoin","Dekado","Lynx","Poseidon Quark","BitcoinWSpectrum","Muse","Trivechain","Dystem","Giant","Peony Coin","Absolute Coin","Vitae","TPCash","ARENON","EUNO","MMOCoin","Ketan","XDNA","PAXEX","ThunderStake","Kcash","Bettex coin","BitMoney","Junson Ming Chan Coin","HerbCoin","PirateCash","Oduwa","Galilel","Crypto Sports","Credit","Dash Platinum","Nasdacoin","Beetle Coin","Titan Coin","Award","Insane Coin","ALAX","LiteDoge","TruckCoin","OrangeCoin","BitstarCoin","NeosCoin","HyperCoin","PinkCoin","AudioCoin","IncaKoin","Piggy Coin","Genstake","XiaoMiCoin","CapriCoin"," ClubCoin","Radium","Creditbit ","OKCash","Lisk","HiCoin","WhiteCoin","FriendshipCoin","Triangles Coin","EOS","Oxycoin","TigerCash","Particl","Nxt","ZEPHYR","Gapcoin","BitcoinPlus"],"TotalCoinSupply":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[288]},"TotalCoinsMined":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[288]},"class":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"selected":{"id":"1115"},"selection_policy":{"id":"1129"}},"id":"1114","type":"ColumnDataSource"},{"attributes":{"source":{"id":"1114"}},"id":"1121","type":"CDSView"},{"attributes":{"fill_color":{"value":"#30a2da"},"hatch_color":{"value":"#30a2da"},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1117","type":"Scatter"},{"attributes":{"data":{"CoinName":["BiblePay","LitecoinCash","Poa Network","Acute Angle Cloud","Waves"],"TotalCoinSupply":{"__ndarray__":"lPYGX5hMdT+8eapDboZLP4RaKOGWizA//Knx0k1iUD8sQxzr4jYaPw==","dtype":"float64","order":"little","shape":[5]},"TotalCoinsMined":{"__ndarray__":"j1gNkdpTXT+QpMXySiFFP5wZJXiPEis/ekNDBbiMUD/7RX5gv3oaPw==","dtype":"float64","order":"little","shape":[5]},"class":[1,1,1,1,1]},"selected":{"id":"1136"},"selection_policy":{"id":"1152"}},"id":"1135","type":"ColumnDataSource"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#e5ae38"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#e5ae38"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1180","type":"Scatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01838","sizing_mode":"stretch_width"},"id":"1351","type":"Spacer"},{"attributes":{},"id":"1115","type":"Selection"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01837","sizing_mode":"stretch_width"},"id":"1071","type":"Spacer"},{"attributes":{},"id":"1182","type":"Selection"},{"attributes":{"below":[{"id":"1087"}],"center":[{"id":"1090"},{"id":"1094"}],"height":300,"left":[{"id":"1091"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"1120"},{"id":"1141"},{"id":"1163"},{"id":"1187"}],"right":[{"id":"1132"}],"sizing_mode":"fixed","title":{"id":"1079"},"toolbar":{"id":"1101"},"width":700,"x_range":{"id":"1072"},"x_scale":{"id":"1083"},"y_range":{"id":"1073"},"y_scale":{"id":"1085"}},"id":"1078","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1112","type":"BasicTickFormatter"},{"attributes":{"fill_color":{"value":"#6d904f"},"hatch_color":{"value":"#6d904f"},"line_color":{"value":"#6d904f"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1184","type":"Scatter"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"1120"}]},"id":"1133","type":"LegendItem"},{"attributes":{"callback":null,"renderers":[{"id":"1120"},{"id":"1141"},{"id":"1163"},{"id":"1187"}],"tags":["hv_created"],"tooltips":[["class","@{class}"],["TotalCoinsMined","@{TotalCoinsMined}"],["TotalCoinSupply","@{TotalCoinSupply}"],["CoinName","@{CoinName}"]]},"id":"1074","type":"HoverTool"},{"attributes":{"coordinates":null,"data_source":{"id":"1181"},"glyph":{"id":"1184"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1186"},"nonselection_glyph":{"id":"1185"},"selection_glyph":{"id":"1206"},"view":{"id":"1188"}},"id":"1187","type":"GlyphRenderer"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#fc4f30"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#fc4f30"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1156","type":"Scatter"},{"attributes":{"end":1.042857142857143,"reset_end":1.042857142857143,"reset_start":-0.04285714285714286,"start":-0.04285714285714286,"tags":[[["TotalCoinsMined","TotalCoinsMined",null]]]},"id":"1072","type":"Range1d"},{"attributes":{"fill_color":{"value":"#e5ae38"},"hatch_color":{"value":"#e5ae38"},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1160","type":"Scatter"},{"attributes":{"label":{"value":"3"},"renderers":[{"id":"1187"}]},"id":"1205","type":"LegendItem"},{"attributes":{"tools":[{"id":"1074"},{"id":"1095"},{"id":"1096"},{"id":"1097"},{"id":"1098"},{"id":"1099"}]},"id":"1101","type":"Toolbar"},{"attributes":{"end":1.1,"reset_end":1.1,"reset_start":-0.1,"start":-0.1,"tags":[[["TotalCoinSupply","TotalCoinSupply",null]]]},"id":"1073","type":"Range1d"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#30a2da"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#30a2da"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1134","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.1},"line_color":{"value":"#6d904f"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1185","type":"Scatter"},{"attributes":{},"id":"1109","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.2},"line_color":{"value":"#6d904f"},"size":{"value":5.477225575051661},"x":{"field":"TotalCoinsMined"},"y":{"field":"TotalCoinSupply"}},"id":"1186","type":"Scatter"},{"attributes":{"source":{"id":"1181"}},"id":"1188","type":"CDSView"}],"root_ids":["1070"]},"title":"Bokeh Application","version":"2.4.2"}};
    var render_items = [{"docid":"620cbc25-7be6-4956-ac3c-2ebc74c57fb4","root_ids":["1070"],"roots":{"1070":"6d07fec8-db25-4ace-9a23-01d7c5335c07"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>




```python

```
