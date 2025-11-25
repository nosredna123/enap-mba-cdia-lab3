# Classificação de solicitações de desastres

Este projeto processa todos os arquivos `.xls` em `relatorios/`, valida que as planilhas compartilham o mesmo esquema e classifica cada linha da coluna **"Solicitação"** usando o modelo `gpt-5-mini`. O resultado gera um arquivo Parquet combinado com uma coluna extra **"Categoria da Solicitação"** em português do Brasil, mantendo cache local para reduzir chamadas à API e reinserindo as classificações em execuções futuras.

## Visão geral do fluxo
1. Carrega prompts em inglês de `prompts/system_prompt.txt` e `prompts/user_prompt_template.txt`. Ambos orientam o modelo a responder em português do Brasil, proibindo invenção de exemplos e exigindo retorno em JSON.
2. Percorre cada arquivo `.xls` na pasta de entrada (padrão: `relatorios/`), lendo todas as planilhas via `pandas`/`xlrd`.
3. Valida de forma fail-fast que todas as planilhas têm os mesmos cabeçalhos e que **"Solicitação"** é a última coluna (sem duplicatas).
4. Normaliza o texto da solicitação; se estiver vazio, a classificação é pulada e a célula de categoria permanece vazia.
5. Constrói o contexto de categorias em português (nome, descrição e até três exemplos reais armazenados em cache) e chama o LLM com retentativas e backoff exponencial.
6. Mescla categorias retornadas pelo modelo, atualiza exemplos com o texto atual e grava as classificações no cache JSON.
7. Junta todas as planilhas processadas em `output/classificacoes_relatorios.parquet` (por padrão) e salva o cache em `cache/classificacao_cache.json`.

## Preparação do ambiente
Opção rápida (recomendada):
```bash
source scripts/setup_env.sh
```
Isso cria `.venv`, ativa e instala `requirements.txt`. Para reativar depois:
```bash
source .venv/bin/activate
```

Opção manual:
1. Crie e ative um ambiente virtual Python 3.10+.
2. Instale dependências com `pip install -r requirements.txt`.

Em ambos os casos, crie `.env` a partir de `.env.example` e preencha `ENAP_LAB3_OPENAI_API_TOKEN` com seu token da API OpenAI.

## Execução
Fluxo objetivo para rodar o processamento:
1) Ative o ambiente virtual (`source .venv/bin/activate` ou `source scripts/setup_env.sh`).
2) Garanta que `relatorios/` contém os `.xls` e que `.env` tem `ENAP_LAB3_OPENAI_API_TOKEN`.
3) Rode o comando abaixo:
```bash
python process_relatorios.py \
  --input-dir relatorios \
  --output-parquet output/classificacoes_relatorios.parquet \
  --cache cache/classificacao_cache.json \
  --system-prompt prompts/system_prompt.txt \
  --user-prompt-template prompts/user_prompt_template.txt \
  --max-retries 3 \
  --backoff-seconds 2.0 \
  --log-level INFO
```

### Opções importantes
- `--input-dir`: diretório com os `.xls` a processar.
- `--output-parquet`: caminho do arquivo Parquet combinado de saída (será criado com diretórios pai).
- `--cache`: arquivo JSON usado para reutilizar classificações e exemplos reais.
- `--max-retries` e `--backoff-seconds`: controlam política genérica de retentativa com exponencial backoff para chamadas ao LLM.
- `--log-level`: ajusta verbosidade (DEBUG para depuração detalhada).

## Comportamento de classificação e cache
- Falhas de esquema ou ausência/duplicação da coluna **"Solicitação"** interrompem a execução imediatamente.
- Textos vazios na coluna-alvo são ignorados sem chamar o modelo.
- A resposta do modelo deve conter `categoria_escolhida` e `categorias_atualizadas` apenas com nome e descrição (sem retorno de exemplos); sempre em português do Brasil.
- As classificações e categorias retornadas são armazenadas no cache; exemplos são deduplicados e mantêm no máximo três amostras por categoria para próximos prompts quando enviados ao modelo.
- Caso o modelo proponha nova categoria, o exemplo inicial é o próprio texto classificado antes de ser salvo em cache.

## Estrutura de arquivos
- `process_relatorios.py`: CLI principal que lê os `.xls`, valida colunas, chama o LLM com cache e monta o Parquet final.
- `llm_client.py`: wrapper da API OpenAI com carregamento opcional de `.env`, estimativa de tokens, extração de texto da resposta e decorador genérico de retentativa com backoff.
- `prompts/`: contém o prompt de sistema e o template de prompt do usuário (ambos em inglês, mas exigindo saídas em português do Brasil).
- `relatorios/`: pasta esperada com os arquivos `.xls` de entrada.
- `output/` e `cache/`: criados automaticamente para o arquivo Parquet combinado e o cache JSON, respectivamente.

## Observações
- O pipeline segue princípios PEP8, DRY e fail-fast para facilitar manutenção e diagnóstico rápido.
- O cache persiste as categorias entre execuções, permitindo que exemplos reais auxiliem novas classificações enquanto minimizam chamadas à API.
