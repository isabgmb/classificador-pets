# Classificador de cães e gatos

Aplicação web que recebe uma foto e responde **cão**, **gato** ou **nem cão nem gato**.

Um modelo próprio (MobileNetV2 com transferência de aprendizado, exportado para TensorFlow Lite)
responde a maioria das fotos. Quando ele fica com confiança baixa, a aplicação consulta o Gemini.
Cada classificação é registrada num banco SQLite, e as fotos duvidosas ficam guardadas para revisão.

```
classificador-pets/
├── treino/                  # roda de vez em quando, de preferência com GPU (Colab)
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

## Atalho: tudo localmente, sem Colab (Windows)

`rodar_tudo.bat` usa sempre o Python do ambiente conda `pets` (e cria o ambiente se ele não
existir), então não cai no Python base do Anaconda. Ele executa `rodar_tudo.py`, que faz as
etapas em ordem e pula as que já estiverem prontas: dependências, dados, treino, avaliação,
`.env` e, por fim, a aplicação, abrindo o navegador em <http://localhost:8000>.

```powershell
.\rodar_tudo.bat --rapido        # primeiro teste: 500 imagens por classe, menos épocas
.\rodar_tudo.bat                 # nas próximas vezes: vai direto para a aplicação
.\rodar_tudo.bat --novo-treino   # treina a próxima versão (v2...) e compara com a anterior
```

Para o treino completo depois do teste rápido, apague a pasta `dados` e rode
`.\rodar_tudo.bat --novo-treino`. Se a nova versão for melhor, troque `MODELO_ATIVO` no `.env`.
Em outros sistemas, rode `python rodar_tudo.py` com os mesmos argumentos, dentro do ambiente.

## 1. Preparar o ambiente

Requer Python 3.11 ou 3.12.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements-treino.txt
```

## 2. Treinar o modelo

Os três comandos abaixo rodam a partir da pasta raiz do projeto. O treino em CPU funciona, mas
é lento; para usar a GPU gratuita do Colab, veja a seção seguinte.

```bash
python treino/preparar_dados.py        # baixa ~1 GB na primeira vez
python treino/treinar.py               # cria modelos/v1/
python treino/avaliar.py --versao v1   # métricas + limiar sugerido
```

A classe "nem cão nem gato" usa a base Imagenette (objetos e cenas). Para o modelo aprender a
separar cães e gatos de **bichos parecidos** (raposas, coelhos, guaxinins), baixe a base
[Animals-10 do Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10), descompacte e
passe a pasta `raw-img`. As pastas de cão e gato dessa base (`cane` e `gatto`) são ignoradas
automaticamente:

```bash
python treino/preparar_dados.py --extra-nenhum caminho/para/raw-img
```

O conjunto de teste é criado uma única vez e não deve mudar entre versões do modelo, senão as
métricas deixam de ser comparáveis. Por isso o script se recusa a sobrescrevê-lo sem `--sobrescrever`.

### Treinar no Google Colab (recomendado)

1. Compacte a pasta do projeto em um `.zip` e envie para o Colab (painel Arquivos).
2. Ative a GPU em **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU**.
3. Rode em uma célula:

```
!unzip -q classificador-pets.zip
%cd classificador-pets
!python treino/preparar_dados.py
!python treino/treinar.py
!python treino/avaliar.py --versao v1
!zip -r v1.zip modelos/v1 -x "*.keras"
```

4. Baixe o `v1.zip` e descompacte dentro da pasta `modelos/` do projeto no seu computador.

## 3. Rodar a aplicação localmente

```bash
cp .env.example .env          # no Windows: copy .env.example .env
uvicorn app.main:criar_app --factory --reload
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
2. Se ela for melhor, mude `MODELO_ATIVO=v2` no `.env` e reinicie a aplicação.
3. Se algo der errado, basta voltar para `MODELO_ATIVO=v1`.

## Próximas etapas

- Página de revisão protegida por login, onde uma pessoa confirma o rótulo das fotos guardadas
  em `dados_app/imagens` (campo `rotulo_confirmado` da tabela `casos`).
- Script que copia as fotos confirmadas para `dados/treino/` antes de um novo treino.
