#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv("insurance.csv") 
data_ds = data.describe() 
print(data_ds)


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
number = ["age", "bmi", "children", "charges"] 
for column in number:
    plt.figure(figsize=(8, 6)) 
    plt.hist(data[column], bins=20, color='yellow', edgecolor='black')
    plt.title(f"Гистрограмма для {column}") 
    plt.xlabel(column) 
    plt.ylabel("Количество") 
    plt.grid(axis='y', linestyle='--', alpha=0.7) 
    plt.show()


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv") 
bmi = data["bmi"] 
charges = data["charges"]
bmi_m = bmi.mean() 
bmi_med = bmi.median() 
charges_m = charges.mean() 
charges_med = charges.median() 
plt.figure(figsize=(12, 5))
plt.hist(bmi, bins=30, color='yellow', alpha=0.7, label='BMI') 
plt.axvline(bmi_med, color='black', linestyle='dashed', linewidth=2, label='Медиана')
plt.title("Гистограмма для индекса массы тела (BMI)") 
plt.xlabel("BMI")
plt.ylabel("Частота") 
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()
plt.figure(figsize=(12, 5))
plt.hist(charges, bins=30, color='red', alpha=0.7, label='Charges') 
plt.axvline(charges_med, color='greenS', linestyle='dashed', linewidth=2, label='Медиана')
plt.title("Гистограмма для расходов (Charges)") 
plt.xlabel("Charges")
plt.ylabel("Частота") 
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()
print("Индекс массы тела (BMI):") 
print(f"Среднее значение: {bmi_m:.2f}")
print(f"Медиана: {bmi_med:.2f}") 
print("\nРасходы (Charges):")
print(f"Среднее значение: {charges_m:.2f}") 
print(f"Медиана: {charges_med:.2f}") 


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
number = ["age", "bmi", "children", "charges"] 
plt.figure(figsize=(12, 8))
for i, column in enumerate(number, 1): 
    plt.subplot(2, 2, i) 
    plt.boxplot(data[column], vert=False) 
    plt.title(f"Box-plot для {column}") 
    plt.xlabel(column)
plt.tight_layout() 
plt.show()


# In[5]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv") 
analyze = "charges" 
num = 300 # выборки
means = [] 
sizes = [10, 30, 50, 100, 200, 500] 
for size in sizes: 
    sample_size = [] 
    for _ in range(num):
        sample = np.random.choice(data[analyze], size= size) 
        mean_s = np.mean(sample) 
        sample_size.append(mean_s)
    means.append(sample_size)
plt.figure(figsize=(15, 10))
for i, size in enumerate(sizes): # построение графиков
    plt.subplot(2, 3, i + 1)
    plt.hist(means[i], bins=30, color='yellow', edgecolor='black') 
    plt.title(f"Размер выборки: {size}")
    plt.xlabel("Cреднее") 
    plt.ylabel("Частота") 
    plt.grid(axis='y')
plt.tight_layout()
plt.show()
for i, size in enumerate(sizes): # вывод отклонений
    mean = np.mean(means[i])
    std_dev = np.std(means[i])
    print(f"Размер выборки: {size}, Среднее: {mean:.2f}, Стандартное отклонение: {std_dev:.2f}")


# In[6]:


import pandas as pd 
import numpy as np
from scipy import stats

data = pd.read_csv("insurance.csv") 
charges = data["charges"] 
bmi = data["bmi"]
confidences = [0.95, 0.99] 
charge_m = np.mean(bmi) #средние значение для массы тел
charge_err = stats.sem(charges) 
bmi_m = np.mean(bmi)
bmi_err = stats.sem(bmi)
charge_interval = [stats.t.interval(confidence_level, len(charges) - 1, loc=charges_m, scale=charge_err) for # подсчёт доверительных интервалов
    confidence_level in confidence_levels]
bmi_interval = [stats.t.interval(confidence_level, len(bmi_data)- 1, loc=bmi_m, scale=std_error_bmi) for
    confidence_level in confidence_levels]
for i, confidence_level in enumerate(confidence_levels): 
    print(f"Доверительный интервал {confidence_level * 100}% для среднего значения расходов: {charge_interval[i]}")
    print(f"Доверительный интервал {confidence_level * 100}% для среднего значения индекса массы тела (bmi): {bmi_interval [i]}")


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("insurance.csv")
bmi = data["bmi"]
charges = data["charges"]
def check(name, data): # Нормальность и построение q-q
    scaler = StandardScaler()
    data_standart = scaler.fit_transform(data.values.reshape(-1, 1))
    statistic, value = stats.kstest(data_standart.flatten(), 'norm')  # тест на нормальность
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(data_standart, bins=20, density=True, alpha=0.6, color='yellow', edgecolor='black')  # Гистограмма
    plt.title(f"Histogram for {name}")
    plt.subplot(1, 2, 2)
    stats.probplot(data_standart.flatten(), plot=plt)
    plt.title(f"Q-Q Plot for {name}")
    print(f"{name}:")
    print(f"KS-статистика: {statistic:.4f}")
    print(f"P-значение: {value:.4f}")
check("BMI", bmi) # проверка на нормальность признаков
check("Charges", charges)
plt.show()


# In[8]:


import pandas as pd
data = pd.read_csv("ECDCCases.csv")

miss = (data.isnull().sum() / len(data)) * 100 # Поиск и вывод пропцщенный значений
print("Процент пропущенных значений:") 
print(miss)
drop = miss.nlargest(2).index # Сортировка по большем пропущенных значений
data.drop(columns=drop, inplace=True) # Удаление
categories = data.select_dtypes(include='object').columns #Категориальные признаки
number = data.select_dtypes(include=['int64', 'float64']).columns #Числовые признаки
data[categories] = data[categories].fillna("other") #Заполнение пропусков
data[number] = data[number].fillna(data[number].median())
miss_after = data.isnull().sum().sum() # Проверка на отсутствие пропусков
print(f"Количество пропусков после обработки: {miss_after}")


# In[9]:


import pandas as pd

data = pd.read_csv("ECDCCases.csv") 
pd.set_option('display.max_columns', None)
number = data.describe() # Статистика по числам
info = data[['deaths']].describe() # Статистика по смертям 
high_death = data[data['deaths'] > 3000] 
high_death_num = len(high_death) 
print("Статистика по признакам:") # Вывод статистики
print(number)
print("Информация о выбросах для 'deaths':") 
print(info)
print(f"Количество дней, когда смертей в день больше 3000:{high_death_num}")
countries = high_death['countriesAndTerritories'].unique() #Стран по смертям
print("Страны, в которых смертей в день больше 3000:") 
for country in countries:
    print(country)


# In[10]:


import pandas as pd

data = pd.read_csv("ECDCCases.csv") 
print(data.info())
duplicates = data[data.duplicated()] 
data = data.drop_duplicates() 
print(data.info())


# In[8]:


import pandas as pd 
from scipy import stats

data = pd.read_csv("bmi.csv")
bmi_north = data[data['region'] == 'northwest']['bmi'] 
bmi_west = data[data['region'] == 'southwest']['bmi'] 
shapiro_north = stats.shapiro(bmi_north) # Проверка нормальности 
shapiro_west = stats.shapiro(bmi_west) # Проверка нормальности 
bartlett = stats.bartlett(bmi_north, bmi_west) # Проверка гомогенности
if shapiro_north.pvalue > 0.05 and shapiro_west.pvalue > 0.05 and bartlett.pvalue > 0.05: # Если обе выборки нормально распределены и гомогенны по дисперсии, выполните t-тест
    stat, value = stats.ttest_ind(bmi_north, bmi_west) 
    print(f"t-статистика: {stat}")
    print(f"p-значение: {value}") 


# In[11]:


import scipy.stats

observe = [97, 98, 109, 95, 97, 104] # Количество выпадений для грани кубика
expect = [100] * 6 # Ожидаемые частоты 
stat, value = scipy.stats.chisquare(observe, expect) # Хи-квадрат
alpha = 0.05 # Уровень значимости
print(f"Статистика критерия Хи-квадрат: {stat:.4f}") 
print(f"P-значение: {value:.4f}")


# In[12]:


import pandas as pd
from scipy.stats import chi2_contingency

data = pd.DataFrame({'Женат': [89, 17, 11, 43, 22, 1],'Гражданский брак': [80, 22, 20, 35, 6, 4],'Не состоит в отношениях': [35, 44, 35, 6, 8, 22]})
data.index = ['Полный рабочий день', 'Частичная занятость', 'Временно не работает', 'На домохозяйстве', 'На пенсии', 'Учёба']
stat, value, _,_  = chi2_contingency(data) # Хи-квадрат
alpha = 0.05 # Уровень значимости
print(f"Статистика критерия Хи-квадрат: {stat}") 
print(f"P-значение: {value}")







