# SIN5014 — BBBC005: contagem automática de núcleos

EACH-USP · SIN5014 — Fundamentos de Processamento Gráfico · 2026

## Estado experimental

Experimentos concluídos em 25/11/2026. 600 pares imagem/máscara da BBBC005 (F1, W2), 24 níveis × 25 amostras. Calibração s01–s05 (120); validação interna retrospectiva s06–s25 (480). Métodos A (N8 sobre a referência, controle privilegiado), B (Watershed original), B* (Watershed calibrado), C0 (Cellpose nuclei, diâmetro fixo 17 px), C1 (diâmetro 60,523633 px) e C* (normalização P0,1–P99,9, análise exploratória pós-hoc). B* = B1, α=0,35, β=0,20, selecionado somente na calibração. 600 controles B5 reproduzidos. A auditoria dos CSVs não é uma nova inferência.

| Método | MAE na validação |
|---|---:|
| A | 36,7750 |
| B | 15,7042 |
| B* | 12,5396 |
| C0 | 24,3250 |
| C1 | 20,4354 |
| C* | 23,6708 |

B* reduziu o MAE em 20,15% em relação a B; ΔMAE −3,164583, IC95% individual [−3,339635; −2,989583], bootstrap pareado por nível (2.000 reamostragens, semente 0). Dice final B* = 0,989504. Não generalizar a superioridade para imagens clínicas ou outros modelos. Dice binário não avalia a separação correta de todas as instâncias.

## Reproduzir e conferir

- `docs/GUIA_REPRODUCAO.md`: comandos, versões e dados necessários.
- `docs/auditoria_bstar_final.json`: auditoria independente dos resultados arquivados.
- `scripts/`: código executado, auditorias e diagnósticos.
- `legacy/`: executor histórico preservado sem alteração.
- `environment/`: ambientes originais.
- `data/manifestos/manifest_f1_w2.csv`: 600 pares com SHA-256.
- `results/`: CSVs e figuras experimentais; não confundir a partição de calibração com a validação.
- `reprodutibilidade/figuras/`: fonte Python, entradas e figuras monocromáticas PT/EN. Regenerar com `python reprodutibilidade/figuras/gerar_figuras_artigo_v06.py` no ambiente clássico; não executa segmentações.
- `docs/README_historico.md`: README inicial, preservado como evidência histórica.

A revisão experimental está identificada pela tag `sin5014-experimentos-v1`, a ser criada pelo script de fechamento. O artigo e suas fontes definitivas serão versionados depois, em uma revisão documental distinta. Para reproduzir, obtenha o código e os dados da tag experimental, não de versões intermediárias dos documentos.

BBBC005: https://bbbc.broadinstitute.org/BBBC005. Cellpose: https://github.com/MouseLand/cellpose. Imagens TIFF, pesos do modelo e ambientes instalados não são redistribuídos neste repositório. Verifique os termos de uso das fontes originais. Não foi atribuída automaticamente uma licença ao código do projeto; a autorização de redistribuição deve ser definida pelo autor.

O ZIP original `entrega_final_20261125_104334.zip` tem SHA-256 `d63f3e19bcf4cb1ce28f6ecdac8780731bc9041d7de4029935b98dea6662a06b`. Está preservado separadamente no backup e no pacote de evidências; não precisa estar dentro do Git. O ZIP e os arquivos de entrada históricos são necessários para o coletor de fechamento, não para a execução dos executores científicos. Os mapas NPY do piloto são preservados em backup próprio; não estão incluídos no Git.
