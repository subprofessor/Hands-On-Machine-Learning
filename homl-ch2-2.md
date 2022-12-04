---
title: HOML_ch2_(2)
description: 머신러닝 전체 흐름 살펴보기 (2)
slug: homl-ch2-2
category: Data-Science
author: Hyunseop Lee
---
# Hands-On Machine Learning

## Ch2. 머신러닝 프로젝트 처음부터 끝까지

### 2.4.2 상관계수 구하기 
<br/>
<br/>
(1) .corr() 메서드를 사용하여 수치적으로 확인



```python
corr_matrix = housing.corr()
```


```python
corr_matrix
```





  <div id="df-907e7d64-abe2-48fc-b4fa-6ea0455a4512">
    <div class="colab-df-container">
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>longitude</th>
      <td>1.000000</td>
      <td>-0.924664</td>
      <td>-0.108197</td>
      <td>0.044568</td>
      <td>0.069608</td>
      <td>0.099773</td>
      <td>0.055310</td>
      <td>-0.015176</td>
      <td>-0.045967</td>
      <td>-0.010690</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>-0.924664</td>
      <td>1.000000</td>
      <td>0.011173</td>
      <td>-0.036100</td>
      <td>-0.066983</td>
      <td>-0.108785</td>
      <td>-0.071035</td>
      <td>-0.079809</td>
      <td>-0.144160</td>
      <td>-0.085528</td>
    </tr>
    <tr>
      <th>housing_median_age</th>
      <td>-0.108197</td>
      <td>0.011173</td>
      <td>1.000000</td>
      <td>-0.361262</td>
      <td>-0.320451</td>
      <td>-0.296244</td>
      <td>-0.302916</td>
      <td>-0.119034</td>
      <td>0.105623</td>
      <td>-0.146920</td>
    </tr>
    <tr>
      <th>total_rooms</th>
      <td>0.044568</td>
      <td>-0.036100</td>
      <td>-0.361262</td>
      <td>1.000000</td>
      <td>0.930380</td>
      <td>0.857126</td>
      <td>0.918484</td>
      <td>0.198050</td>
      <td>0.134153</td>
      <td>0.220528</td>
    </tr>
    <tr>
      <th>total_bedrooms</th>
      <td>0.069608</td>
      <td>-0.066983</td>
      <td>-0.320451</td>
      <td>0.930380</td>
      <td>1.000000</td>
      <td>0.877747</td>
      <td>0.979728</td>
      <td>-0.007723</td>
      <td>0.049686</td>
      <td>0.015662</td>
    </tr>
    <tr>
      <th>population</th>
      <td>0.099773</td>
      <td>-0.108785</td>
      <td>-0.296244</td>
      <td>0.857126</td>
      <td>0.877747</td>
      <td>1.000000</td>
      <td>0.907222</td>
      <td>0.004834</td>
      <td>-0.024650</td>
      <td>0.025809</td>
    </tr>
    <tr>
      <th>households</th>
      <td>0.055310</td>
      <td>-0.071035</td>
      <td>-0.302916</td>
      <td>0.918484</td>
      <td>0.979728</td>
      <td>0.907222</td>
      <td>1.000000</td>
      <td>0.013033</td>
      <td>0.065843</td>
      <td>0.038490</td>
    </tr>
    <tr>
      <th>median_income</th>
      <td>-0.015176</td>
      <td>-0.079809</td>
      <td>-0.119034</td>
      <td>0.198050</td>
      <td>-0.007723</td>
      <td>0.004834</td>
      <td>0.013033</td>
      <td>1.000000</td>
      <td>0.688075</td>
      <td>0.902750</td>
    </tr>
    <tr>
      <th>median_house_value</th>
      <td>-0.045967</td>
      <td>-0.144160</td>
      <td>0.105623</td>
      <td>0.134153</td>
      <td>0.049686</td>
      <td>-0.024650</td>
      <td>0.065843</td>
      <td>0.688075</td>
      <td>1.000000</td>
      <td>0.643892</td>
    </tr>
    <tr>
      <th>income_cat</th>
      <td>-0.010690</td>
      <td>-0.085528</td>
      <td>-0.146920</td>
      <td>0.220528</td>
      <td>0.015662</td>
      <td>0.025809</td>
      <td>0.038490</td>
      <td>0.902750</td>
      <td>0.643892</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-907e7d64-abe2-48fc-b4fa-6ea0455a4512')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-907e7d64-abe2-48fc-b4fa-6ea0455a4512 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-907e7d64-abe2-48fc-b4fa-6ea0455a4512');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
corr_matrix["median_house_value"]   #중간 주택 가격 열을 불러옴으로 위도, 경도 등 다른 특성 사이의 상관관계를 확인
```




    longitude            -0.045967
    latitude             -0.144160
    housing_median_age    0.105623
    total_rooms           0.134153
    total_bedrooms        0.049686
    population           -0.024650
    households            0.065843
    median_income         0.688075
    median_house_value    1.000000
    income_cat            0.643892
    Name: median_house_value, dtype: float64




```python
corr_matrix["median_house_value"].sort_values(ascending=False)   #sort_values()를 사용하여 출력값 정렬. ascending=False는 내림차순을 의미 
```




    median_house_value    1.000000
    median_income         0.688075
    income_cat            0.643892
    total_rooms           0.134153
    housing_median_age    0.105623
    households            0.065843
    total_bedrooms        0.049686
    population           -0.024650
    longitude            -0.045967
    latitude             -0.144160
    Name: median_house_value, dtype: float64



* 상관계수는 선형적인 상관관계만 측정합니다. 비선형적인 관계는 잡을 수 없으며 기울기에 관계가 없습니다.

(2) scatter_matrix 함수로 산점도 행렬 구하기


```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]  #housing DataFrame의 특성이 11개 이므로 상관계수는  총 11*2 = 121개가 그려짐. 몇 개를 선택한 것. attributes 리스트를 사용하지 않고 바로 넣어도 가능
scatter_matrix(housing[attributes], figsize=(12,12))   #figsize=(가로, 세로)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7fe95fb65b90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95ff8c790>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95f2e32d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95f940b50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fe95f8e2fd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95f2ef3d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95fe8c2d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95fca6290>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fe95fca6990>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95fd87550>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95fc95310>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95fbb3a50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fe95fbd9350>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95fc46e10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe95f967910>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fe960041ed0>]],
          dtype=object)




    
![png](homl-ch2-1_files/homl-ch2-1_55_1.png)
    


대각요소(diagonal entry)들은 자신에 대한 상관계수이기 때문에 직선이 되어버립니다. 의미가 없는 결과이기 때문에 pandas에서는 각 특성의 히스토그램을 출력합니다.


```python
housing.plot(kind="scatter", x="median_income",y="median_house_value",alpha=0.1)   #median_income과 median_house_value 데이터를 받아와서 산점도를 그린 것
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe95ff22d10>




    
![png](homl-ch2-1_files/homl-ch2-1_57_1.png)
    


위 그래프가 시사하는 것들<br/>

* 상관관계가 매우 강하다. 위쪽으로 향하는 경향을 볼 수 있으며 포인트들이 많이 흩어져 있지 않다.<br/>

* 앞서 본 가격 제한 값이 &#36;500,000에서 수평선으로 잘 보이지만 그래프의 형태를 망가뜨린다. <br/>
* 이러한 수평선은 &#36;280,000, &#36;350,000, &#36;450,000 근처에도 있는데 학습 측면에서는 그리 좋은 데이터가 아니므로 해당 구역을 제거하는 것이 좋다.
