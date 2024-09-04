import pandas as pd
import matplotlib.pyplot as plt

# Učitaj prikupljene podatke
df = pd.read_csv('game_data.csv')

# Vizualiziraj distribuciju nagrada
plt.figure(figsize=(10, 5))
plt.hist(df['reward'], bins=20, color='blue', edgecolor='black')
plt.title('Distribucija Nagrada')
plt.xlabel('Nagrada')
plt.ylabel('Frekvencija')
plt.show()

# Analiza korelacija
correlations = df.corr()
print("Korelacije između značajki:")
print(correlations)

# Vizualizacija korelacija
plt.figure(figsize=(8, 6))
plt.imshow(correlations, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlations)), correlations.columns, rotation=90)
plt.yticks(range(len(correlations)), correlations.columns)
plt.title('Korelacijska matrica')
plt.show()
