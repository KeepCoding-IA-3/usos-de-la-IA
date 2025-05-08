from datasets import load_dataset
import pandas as pd

dataset = load_dataset("imdb")
print(dataset.keys())

print("\nPrimeras 5 reseñas de entrenamiento:")
print(dataset["train"][:5])

# 🔄 Convierte a DataFrame para analizarlo con Pandas
df_train = pd.DataFrame(dataset["train"][:500])  # puedes cambiar 500 por más
print("\nEstructura del DataFrame:")
print(df_train.head())

print("\nDistribución de sentimientos (0 = Negativo, 1 = Positivo):")
print(df_train["label"].value_counts())