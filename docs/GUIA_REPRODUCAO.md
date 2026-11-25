# SIN5014 — Guia de reprodução experimental

Consulte o README para os resultados finais e `docs/auditoria_bstar_final.json` para a auditoria. A tag científica fixa é `sin5014-experimentos-v1`. Os PDFs e fontes LaTeX finais serão incluídos posteriormente, em uma revisão documental separada.

## Reproduzir os gráficos

No ambiente clássico, execute `python reprodutibilidade/figuras/gerar_figuras_artigo_v06.py`. As entradas e figuras originais estão na mesma pasta; a saída padrão é `figuras_regeneradas`. Nenhuma segmentação é executada. Os PDFs podem ter metadados de compilação variáveis.

## Reproduzir os experimentos (opcional)
Extraia o código e os resultados em uma pasta dedicada (por exemplo, SIN5014-BBBC005). Não sobrescreva os originais. Extraia os dados da BBBC005 em uma pasta separada, com a mesma estrutura relativa do manifesto. Dados: https://bbbc.broadinstitute.org/BBBC005. Utilize o canal w2, foco F1, 600 pares. O manifesto verifica os bytes de cada TIFF, não basta coincidir o nome. Preserve a versão dos pesos `nucleitorch_0` (SHA-256 89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303).

Os comandos abaixo são exemplos para PowerShell. Use uma nova sessão e ajuste os caminhos às pastas extraídas. As pastas de saída novas devem estar ausentes. Cada linha inclui a identificação da rotina de reprodução (não são passos operacionais do fechamento atual).

```powershell
$Repo = (Get-Location).Path # SIN5014 - REPRODUCAO - PASSO R01
$Base = 'CAMINHO_DA_BBBC005' # SIN5014 - REPRODUCAO - PASSO R02
$Manifest = Join-Path $Repo 'data/manifestos/manifest_f1_w2.csv' # SIN5014 - REPRODUCAO - PASSO R03
$B = Join-Path $Repo 'results/reproducao_v04_20261124_201858/por_imagem.csv' # SIN5014 - REPRODUCAO - PASSO R04
$CP = Join-Path $Repo 'results/cellpose_completo_20261124_214400' # SIN5014 - REPRODUCAO - PASSO R05
```

Ambiente clássico: Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0, Pandas 3.1.0, Matplotlib 3.11.2, tifffile conforme environment/conda_classic_windows.txt. Use o registro do ambiente como inventário; caminhos absolutos e builds específicos devem ser ajustados a cada máquina. O comportamento observado da remoção de pequenos objetos é área <=20, com conectividade 4.

```powershell
python scripts/gerar_manifesto_bbbc005.py --base "$Base" --out "$Manifest" --verify # SIN5014 - REPRODUCAO - PASSO R06
python legacy/experimento_bbbc005.py --images "$Base/imagens/BBBC005_v1_images" --masks "$Base/ground_truth/BBBC005_v1_ground_truth" --out "results/reproducao_nova" # SIN5014 - REPRODUCAO - PASSO R07
python scripts/executar_watershed_bstar_v1.py --base "$Base" --manifest "$Manifest" --baseline "$B" --out "results/calibracao_nova" # SIN5014 - REPRODUCAO - PASSO R08
```

A reprodução clássica usa o código histórico preservado. O comando R07 pressupõe a estrutura de diretórios original; confira o manifesto se a organização for diferente. O programa de calibração verifica a versão das bibliotecas e o baseline individual. Para auditar os resultados históricos, use `scripts/auditar_reproducao_bbbc005.py --help` (o argumento `--legacy` refere-se a `legacy/experimento_bbbc005.py`, cujo SHA-256 é verificado). Consulte o protocolo metodológico em `docs/`.

Ambiente Cellpose: Python 3.11.18, Cellpose 3.1.1.2, PyTorch 2.7.1+cpu, NumPy 1.26.4, SciPy 1.17.1 e demais versões registradas em `environment/requirements_cellpose_cpu.txt`. O arquivo de requisitos é o inventário original; entradas `file:///` não são portáveis. Instale o PyTorch CPU pelo índice oficial compatível e verifique versões e pesos. Repositório do modelo: https://github.com/MouseLand/cellpose. Não use a versão mais recente indiscriminadamente.

```powershell
python scripts/executar_cellpose_bbbc005_v1.py --base "$Base" --manifest "$Manifest" --out "results/cellpose_reexecucao" # SIN5014 - REPRODUCAO - PASSO R09
python scripts/testar_normalizacao_cellpose_v2.py --base "$Base" --manifest "$Manifest" --baseline-dir "$CP" --out "results/piloto_normalizacao_novo" # SIN5014 - REPRODUCAO - PASSO R10
python scripts/executar_cellpose_cstar_v1.py --base "$Base" --manifest "$Manifest" --baseline-dir "$CP" --partition calibracao --out "results/cstar_cal_nova" # SIN5014 - REPRODUCAO - PASSO R11
python scripts/executar_cellpose_cstar_v1.py --base "$Base" --manifest "$Manifest" --baseline-dir "$CP" --partition validacao --out "results/cstar_val_nova" # SIN5014 - REPRODUCAO - PASSO R12
python scripts/diagnosticar_percentis_por_nivel_bbbc005.py --base "$Base" --manifest "$Manifest" --out "results/percentis_novo" # SIN5014 - REPRODUCAO - PASSO R13
```

Os executores C* e final exigem hashes históricos específicos de entrada. Utilize os CSVs e protocolos originais fornecidos no ZIP de resultados, não os CSVs de uma nova execução (tempos e metadados podem mudar). O programa final consome as pastas históricas nomeadas no código. É possível regenerar a validação B*/B5 e consolidar os seis métodos sem repetir Cellpose:

```powershell
conda activate 'D:/LabCV/ambientes/visao-python' # SIN5014 - REPRODUCAO - PASSO R14
python scripts/fechar_experimentos_sin5014.py --repo "$Repo" --base "$Base" --cal "$Repo/results/watershed_bstar_calibracao_20261125_095745" --out "$Repo/results/entrega_reproducao_nova" # SIN5014 - REPRODUCAO - PASSO R15
```

Resultado esperado: 480 controles B5 na validação e 3.600 linhas consolidadas (600 por método). MAE B=15,7041667 e B*=12,5395833 na validação. Não exigir hashes idênticos dos arquivos recém-gerados quando houver tempos ou metadados variáveis; preserve e verifique o pacote original. Não alterar a grade ou selecionar parâmetros usando validação.


## Git

Use a tag científica `sin5014-experimentos-v1`. Confirme o endereço remoto efetivo antes de fazer clone. Não distribua dados ou pesos sem conferir suas licenças.
