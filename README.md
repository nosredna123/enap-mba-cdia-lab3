# Extração de objetos de solicitações de desastres

Este projeto processa todos os arquivos `.xls` em `relatorios/`, valida que as planilhas compartilham o mesmo esquema e extrai o objeto específico de cada linha da coluna **"Solicitação"** usando o modelo `gpt-5-mini`. O resultado gera um arquivo Parquet combinado com uma coluna extra **"Objeto da Solicitação"** em português do Brasil, mantendo cache local para reduzir chamadas à API e reinserindo os objetos extraídos em execuções futuras.

## Visão geral do fluxo
1. Carrega prompts em inglês de `prompts/system_prompt.txt` e `prompts/user_prompt_template.txt`. Ambos orientam o modelo a responder em português do Brasil, identificando objetos físicos concretos (ex: pontes, pavimentação, unidades habitacionais) e exigindo retorno em JSON.
2. Percorre cada arquivo `.xls` na pasta de entrada (padrão: `relatorios/`), lendo todas as planilhas via `pandas`/`xlrd`.
3. Valida de forma fail-fast que todas as planilhas têm os mesmos cabeçalhos e que **"Solicitação"** é a última coluna (sem duplicatas).
4. Normaliza o texto da solicitação; se estiver vazio, a extração é pulada e a célula de objeto permanece vazia.
5. Constrói o contexto de objetos em português (nome, descrição e até três exemplos reais armazenados no registro de objetos) e chama o LLM com retentativas e backoff exponencial.
6. Após cada nova extração, persiste imediatamente:
   - O mapeamento texto→objeto em `cache/objetos_cache.json`
   - As definições de objetos atualizadas em `cache/objetos_registry.json`
7. Junta todas as planilhas processadas em `output/classificacoes_relatorios.parquet` (por padrão).

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
  --object-cache cache/objetos_cache.json \
  --objects-registry cache/objetos_registry.json \
  --system-prompt prompts/system_prompt.txt \
  --user-prompt-template prompts/user_prompt_template.txt \
  --max-retries 3 \
  --backoff-seconds 2.0 \
  --log-level INFO
```

### Opções importantes
- `--input-dir`: diretório com os `.xls` a processar.
- `--output-parquet`: caminho do arquivo Parquet combinado de saída (será criado com diretórios pai).
- `--object-cache`: arquivo JSON que armazena mapeamentos texto→objeto (leve e rápido).
- `--objects-registry`: arquivo JSON que mantém as definições de objetos com descrições e exemplos.
- `--max-retries` e `--backoff-seconds`: controlam política genérica de retentativa com exponencial backoff para chamadas ao LLM.
- `--log-level`: ajusta verbosidade (DEBUG para depuração detalhada).

## Comportamento de extração de objetos e cache
- Falhas de esquema ou ausência/duplicação da coluna **"Solicitação"** interrompem a execução imediatamente.
- Textos vazios na coluna-alvo são ignorados sem chamar o modelo.
- A resposta do modelo deve conter `objeto_escolhido` e `objetos_atualizados` apenas com nome e descrição (sem retorno de exemplos); sempre em português do Brasil.
- **Objetos base pré-definidos**: pontes, bueiros e galerias, pavimentação, unidades habitacionais, edificações/prédios públicos.
- **Sistema de cache otimizado em dois arquivos**:
  - `objetos_cache.json`: armazena apenas o mapeamento texto→objeto (leve, escrita rápida)
  - `objetos_registry.json`: mantém as definições completas de objetos com descrições e exemplos
- Após cada nova extração, ambos os arquivos são persistidos imediatamente (fail-fast), prevenindo perda de dados e permitindo interrupção/retomada do processamento.
- Exemplos são deduplicados e mantêm no máximo três amostras por objeto para próximos prompts ao modelo.
- Caso o modelo proponha novo objeto, o exemplo inicial é o próprio texto analisado.
- O modelo é orientado a ser conservador ao criar novos objetos, evitando redundâncias (ex: "pontilhões" deve ser mapeado para "pontes").

## Estrutura de arquivos
- `process_relatorios.py`: CLI principal que lê os `.xls`, valida colunas, chama o LLM com cache e monta o Parquet final.
- `llm_client.py`: wrapper da API OpenAI com carregamento opcional de `.env`, estimativa de tokens, extração de texto da resposta e decorador genérico de retentativa com backoff.
- `prompts/`: contém o prompt de sistema e o template de prompt do usuário (orientam o modelo a identificar objetos físicos concretos em português do Brasil).
- `relatorios/`: pasta esperada com os arquivos `.xls` de entrada.
- `cache/`: criado automaticamente para armazenar arquivos JSON e JSONL:
  - `objetos_cache.json`: mapeamentos leves texto→objeto
  - `objetos_registry.json`: registro completo de objetos com descrições e exemplos
  - `reasoning_objetos.jsonl`: logs de raciocínio das extrações individuais
  - `reasoning_objetos_registry.jsonl`: logs de raciocínio das atualizações do registro
- `output/`: criado automaticamente para o arquivo Parquet e CSV combinados.

## Observações
- O pipeline segue princípios PEP8, DRY e fail-fast para facilitar manutenção e diagnóstico rápido.
- O sistema de cache otimizado elimina redundância, reduzindo drasticamente o tamanho dos arquivos e o tempo de I/O.
- Cache é persistido após cada extração, prevenindo perda de progresso em caso de interrupção.
- O foco está em identificar **objetos físicos concretos** (infraestrutura, edificações) mencionados nas solicitações, evitando abstrações ou categorizações amplas.
