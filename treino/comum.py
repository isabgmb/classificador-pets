"""Constantes compartilhadas pelos scripts de treino."""

SEED = 42

# Ordem fixa das classes: o índice de cada uma é a posição na lista.
# A aplicação lê essa mesma ordem do arquivo metadados.json de cada modelo.
CLASSES = ["cao", "gato", "nenhum"]
NOMES = {"cao": "Cão", "gato": "Gato", "nenhum": "Nem cão nem gato"}

CONJUNTOS = ["treino", "validacao", "teste"]
