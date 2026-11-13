# Classificador de cães e gatos

Aplicação web que recebe uma foto e responde **cão**, **gato** ou **nem cão nem gato**.

Um modelo próprio (MobileNetV2 com transferência de aprendizado, exportado para TensorFlow Lite)
responde a maioria das fotos. Quando ele fica com confiança baixa, a aplicação consulta o Gemini.
Cada classificação é registrada num banco SQLite, e as fotos duvidosas ficam guardadas para revisão.

```
classificador-pets/
├── rodar_tudo.bat           # atalho para Windows: roda tudo com o Python do ambiente "pets"
├── rodar_tudo.py            # instala, prepara os dados, treina, avalia e sobe a aplicação
├── treino/                  # roda de vez em quando (no computador ou no Colab)
│   ├── preparar_dados.py    # monta a base com 3 classes e divide 70/15/15
│   ├── treinar.py           # treina e exporta modelos/vN/modelo.tflite
│   └── avaliar.py           # métricas no teste fixo e limiar de confiança
├── app/                     # a aplicação, só faz previsão
│   ├── main.py              # API (FastAPI) e fluxo híbrido
│   ├── classificador.py     # carrega o .tflite e prevê
│   ├── gemini.py            # consulta ao Gemini com resposta em formato fixo
│   ├── registro.py          # banco de casos para revisão e retreino
│   ├── padroes.md           # diretrizes curtas, revisadas por pessoas, enviadas ao Gemini
│   └── static/index.html    # página de envio de fotos
├── modelos/                 # uma pasta por versão: v1/, v2/...
└── tests/                   # testes automáticos (não chamam a API do Gemini)
```

## Como o modelo é treinado

- **Arquitetura:** MobileNetV2 pré-treinada no ImageNet, com uma nova camada de saída para as
  3 classes (transferência de aprendizado).
- **Treino em duas fases:** primeiro com a base congelada, treinando só a camada nova; depois,
  ajuste fino das últimas 30 camadas da base, com taxa de aprendizado bem menor.
- **Exportação:** TensorFlow Lite com quantização de pesos (cerca de 2,5 MB), que é o arquivo
  usado pela aplicação.
- **Gemini:** não participa do treino. Ele só é consultado pela aplicação quando a confiança do
  modelo fica abaixo do limiar.

## Atalho: tudo localmente, sem Colab (Windows)

`rodar_tudo.bat` usa sempre o Python do ambiente conda `pets` (e cria o ambiente com Python 3.12
se ele não existir), então não cai no Python base do Anaconda. Ele executa `rodar_tudo.py`, que
faz as etapas em ordem e **pula as que já estiverem prontas**:

1. confere o Python e instala as dependências que faltarem;
2. baixa e prepara as imagens (pula se a pasta `dados/` já existir);
3. treina o modelo (pula se já houver um `modelos/vN/modelo.tflite`);
4. avalia o modelo e calcula o limiar;
5. cria o `.env` a partir do `.env.example`, apontando para a versão treinada;
6. sobe a aplicação e abre o navegador em <http://localhost:8000> (Ctrl+C encerra).

No PowerShell, dentro da pasta do projeto (ou com dois cliques no `.bat`):

```powershell
.\rodar_tudo.bat --rapido        # primeiro teste: 500 imagens por classe e menos épocas
.\rodar_tudo.bat                 # nas próximas vezes: vai direto para a aplicação
.\rodar_tudo.bat --novo-treino   # treina a próxima versão (v2...) e compara com a anterior
```

| Opção | O que faz |
|---|---|
| `--rapido` | 500 imagens por classe e 4 + 3 épocas, para um primeiro teste |
| `--novo-treino` | treina uma nova versão mesmo que já exista uma |
| `--sem-download` | não baixa a Imagenette; gera imagens artificiais para "nem cão nem gato" |
| `--extra-nenhum PASTA` | acrescenta fotos de outros animais à classe "nem cão nem gato" |
| `--sem-app` | para depois da avaliação, sem subir o servidor |

**Tempo esperado:** os downloads somam cerca de 1,1 GB (base de cães e gatos ~790 MB e
Imagenette ~325 MB) e dependem da internet. Sem GPU, o treino rápido leva poucos minutos e o
completo, de 30 minutos a mais de 1 hora, dependendo do processador.

> **Atenção com `--sem-download`:** serve só para ver a aplicação funcionando quando a internet
> está lenta. As imagens artificiais (degradês, formas e ruído) são fáceis demais de separar, então
> as métricas ficam otimistas (perto de 100%) e fotos reais sem cão nem gato tendem a sair como
> cão ou gato. Para o trabalho e a apresentação, treine com a base completa: apague a pasta
> `dados` e rode `.\rodar_tudo.bat --novo-treino`. Se a nova versão for melhor, troque
> `MODELO_ATIVO` no `.env`.

Em outros sistemas, rode `python rodar_tudo.py` com os mesmos argumentos, dentro do ambiente.

## 1. Preparar o ambiente

Requer Python 3.11 ou 3.12. Use um ambiente separado só para este projeto, para não misturar
com pacotes de outros projetos (no Python base do Anaconda isso causa conflitos de `numpy`,
`h5py` e `protobuf`).

**Com Anaconda (Windows):**

```powershell
conda create -n pets python=3.12 -y
conda activate pets
python -c "import sys; print(sys.executable)"   # precisa mostrar ...\anaconda3\envs\pets\python.exe
python -m pip install -r requirements-treino.txt
```

Se o `conda activate` não fizer efeito no PowerShell (o caminho mostrado continua sendo
`anaconda3\python.exe`), chame o Python do ambiente pelo caminho completo, sem ativar nada:

```powershell
$py = "C:\Users\<seu-usuario>\anaconda3\envs\pets\python.exe"
& $py -m pip install -r requirements-treino.txt
```

Daí em diante, use `& $py` no lugar de `python` nos comandos abaixo. O `$py` vale só para o
terminal atual.

**Com venv (Linux/macOS ou Windows sem Anaconda):**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements-treino.txt
```

## 2. Treinar o modelo

Os três comandos abaixo rodam a partir da pasta raiz do projeto (o `rodar_tudo.py` faz o mesmo
automaticamente). O treino em CPU funciona, mas é mais lento; para usar a GPU gratuita do Colab,
veja a seção seguinte.

```bash
python treino/preparar_dados.py        # baixa ~1,1 GB na primeira vez
python treino/treinar.py               # cria modelos/v1/
python treino/avaliar.py --versao v1   # métricas + limiar sugerido
```

### De onde vêm as imagens

- **Cão e gato:** zip original da base Asirra / cats_vs_dogs (Microsoft Research), lido
  diretamente pelo script. Se o TensorFlow Datasets já tiver baixado esse zip
  (`~/tensorflow_datasets/downloads/cats_vs_dogs`), ele é reaproveitado; senão, é baixado para
  `~/.cache/classificador-pets`. O gerador do TFDS para essa base não é usado porque falha no
  Windows. Os arquivos corrompidos da base original são ignorados.
- **Nem cão nem gato:** base Imagenette (objetos e cenas) pelo TensorFlow Datasets, sem a classe
  de cachorro que ela contém.

Para o modelo aprender a separar cães e gatos de **bichos parecidos** (raposas, coelhos,
guaxinins), baixe a base
[Animals-10 do Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10), descompacte e
passe a pasta `raw-img`. As pastas de cão e gato dessa base (`cane` e `gatto`) são ignoradas
automaticamente:

```bash
python treino/preparar_dados.py --extra-nenhum caminho/para/raw-img
```

Para testar sem baixar a Imagenette, use `--nenhum-sintetico`, que gera imagens artificiais para
a classe "nem cão nem gato". Veja o aviso na seção do atalho: serve só para teste.

O conjunto de teste é criado uma única vez e não deve mudar entre versões do modelo, senão as
métricas deixam de ser comparáveis. Por isso o script se recusa a sobrescrevê-lo sem `--sobrescrever`.

### Treinar no Google Colab (opcional, mais rápido com GPU)

1. Compacte a pasta do projeto em um `.zip` e envie para o Colab (painel Arquivos).
2. Ative a GPU em **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU**.
3. Rode em uma **célula do notebook** (os comandos com `!` são do Colab e não funcionam no PowerShell):

```
!unzip -q classificador-pets.zip
%cd classificador-pets
!python treino/preparar_dados.py
!python treino/treinar.py
!python treino/avaliar.py --versao v1
!zip -r v1.zip modelos/v1 -x "*.keras"
```

4. Baixe o `v1.zip` e descompacte na pasta raiz do projeto (o zip já contém o caminho
   `modelos/v1`). Confira se o arquivo ficou em `modelos/v1/modelo.tflite`.

## 3. Rodar a aplicação localmente

```bash
cp .env.example .env          # no Windows: copy .env.example .env
python -m uvicorn app.main:criar_app --factory --reload
```

Abra <http://localhost:8000>. A documentação interativa da API fica em <http://localhost:8000/docs>.

Sem `GEMINI_API_KEY` no `.env`, a aplicação funciona só com o modelo próprio e mostra um aviso nas
fotos duvidosas. Para ativar o Gemini, gere uma chave em <https://aistudio.google.com/apikey> e
coloque no `.env`. No plano gratuito do Gemini, o Google pode usar as fotos enviadas para melhorar
seus produtos; para fotos de usuários reais, ative o faturamento.

### Rotas

| Rota | O que faz |
|---|---|
| `GET /` | página de envio de fotos |
| `POST /api/classificar` | recebe o campo `foto` (JPEG, PNG ou WebP) e devolve a classificação |
| `GET /api/saude` | versão do modelo, limiar, Gemini ativo e contagem de casos |

Exemplo de resposta de `/api/classificar`:

```json
{
  "id": 12,
  "classe": "gato",
  "rotulo": "Gato",
  "confianca": 0.93,
  "fonte": "gemini",
  "aviso": null,
  "modelo": {"versao": "v1", "classe": "cao", "confianca": 0.61,
             "probabilidades": {"cao": 0.61, "gato": 0.36, "nenhum": 0.03}},
  "gemini": {"classe": "gato", "confianca": 0.93},
  "limiar": 0.87
}
```

`fonte` indica quem respondeu: `modelo` (confiança acima do limiar), `gemini` (modelo em dúvida,
Gemini consultado) ou `modelo_sem_gemini` (modelo em dúvida e Gemini indisponível ou desativado).

## 4. Rodar os testes

```bash
python -m pytest
```

Os testes criam um modelo TFLite minúsculo e substituem o Gemini por uma resposta falsa, então
rodam sem a base de dados, sem o modelo treinado e sem gastar crédito da API.

## 5. Publicar com Docker

```bash
docker build -t classificador-pets .
docker run -p 8000:8000 --env-file .env classificador-pets
```

A mesma imagem pode ser publicada no Google Cloud Run ou no Render. Configure a chave do Gemini
como variável secreta do serviço, nunca dentro da imagem.

## 6. Trocar a versão do modelo

1. Treine e avalie a nova versão comparando com a atual:
   `python treino/avaliar.py --versao v2 --comparar-com v1`
   (o `rodar_tudo.bat --novo-treino` já faz essa comparação).
2. Se ela for melhor, mude `MODELO_ATIVO=v2` no `.env` e reinicie a aplicação.
3. Se algo der errado, basta voltar para `MODELO_ATIVO=v1`.

## Problemas comuns

| Mensagem | Causa e solução |
|---|---|
| Caminhos com `anaconda3\Lib\...` no erro, ou erro de `numpy`/`h5py` | O comando rodou no Python base do Anaconda, não no ambiente `pets`. Use o `rodar_tudo.bat` ou o `& $py` da seção 1. |
| `No module named 'importlib_resources'` | Dependência do TensorFlow Datasets que nem sempre vem junto. Já está no `requirements-treino.txt`; rode o `pip install -r requirements-treino.txt` de novo. |
| `There is no item named 'PetImages\Cat\0.jpg' in the archive` | Defeito do TensorFlow Datasets no Windows. A versão atual do `preparar_dados.py` não usa mais esse gerador; atualize o arquivo. |
| `Modelo não encontrado em ...\modelos\v1` | Ainda não há modelo treinado. Rode o treino (seção 2 ou `rodar_tudo.bat`) ou ajuste `MODELO_ATIVO` no `.env`. |
| `A pasta 'dados/teste' já existe` | Proteção do conjunto de teste. Para refazer os dados, apague a pasta `dados` ou use `--sobrescrever`. |
| `GPU support is not available on native Windows` | Aviso normal: no Windows o TensorFlow treina em CPU. Para usar GPU, treine no Colab. |

## Próximas etapas

- Página de revisão protegida por login, onde uma pessoa confirma o rótulo das fotos guardadas
  em `dados_app/imagens` (campo `rotulo_confirmado` da tabela `casos`).
- Script que copia as fotos confirmadas para `dados/treino/` antes de um novo treino.
