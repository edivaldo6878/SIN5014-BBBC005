
# Protocolo de sensibilidade do Watershed Bstar

Data: 2026-11-25

## 1. Objetivo

Investigar a influencia da distancia minima entre
marcadores e do limiar relativo dos maximos locais
sobre a contagem automatica de nucleos na BBBC005.

## 2. Configuracoes experimentais

Config | Alpha | Beta
B1     | 0.35  | 0.20
B2     | 0.35  | 0.30
B3     | 0.35  | 0.40
B4     | 0.50  | 0.20
B5     | 0.50  | 0.30
B6     | 0.50  | 0.40
B7     | 0.65  | 0.20
B8     | 0.65  | 0.30
B9     | 0.65  | 0.40

B5 corresponde ao metodo B original.

## 3. Operacoes preservadas

- Filtro de mediana 3x3.
- Limiarizacao de Otsu.
- Remocao de componentes com area <= 20 pixels.
- Conectividade N4 na remocao de objetos pequenos.
- Rotulacao de componentes conexos N8.
- Raio equivalente baseado na mediana das areas.
- Transformada de distancia euclidiana.
- Maximos locais e Watershed com marcadores.

A semantica morfologica corresponde ao comportamento
observado no scikit-image 0.26.0.

## 4. Ambiente experimental

NumPy: 2.5.4
SciPy: 1.18.1
scikit-image: 0.26.0
Processamento: CPU

## 5. Particoes

Calibracao: 120 imagens, amostras s01 a s05.
Validacao interna: 480 imagens, amostras s06 a s25.

Calibracao: 9 configuracoes x 120 imagens = 1080.
Validacao: configuracao selecionada e controle B5.

A avaliacao possui carater retrospectivo.

## 6. Criterio de selecao

Selecionar a configuracao com menor soma dos erros
absolutos nas 120 imagens de calibracao.

Em caso de empate, priorizar B5 quando presente
entre os empatados. Caso contrario, selecionar
a configuracao de menor indice.

Nao alterar a grade nem os criterios de selecao
apos conhecer os resultados.

Nao utilizar a validacao para ajustar parametros.

## 7. Controle de reproducao

B5 devera reproduzir as contagens individuais,
Dice e IoU historicos antes da selecao de Bstar.

Divergencias deverao interromper o experimento
para investigacao.

## 8. Metricas

- MAE.
- RMSE.
- MAPE.
- Vies.
- Contagens exatas.
- Subcontagens e supercontagens.
- Dice e IoU anteriores ao Watershed.
- Dice e IoU da uniao dos rotulos finais.
- Precisao e revocacao em pixels.
- Areas dos componentes conexos.
- Raio estimado.
- Distancia minima entre marcadores.
- Limiar efetivo dos maximos locais.
- Quantidade de marcadores.
- Componentes sem marcadores.
- Cobertura da mascara final.

## 9. Analise estatistica

Comparacao principal: Bstar versus B.

Bootstrap pareado estratificado pelos 24 niveis.
Numero de reamostragens: 2000.
Semente: 0.
Intervalo de confianca: 95%.

Preservar resultados favoraveis e desfavoraveis.

## 10. Limitacoes

As mascaras binarias de referencia nao permitem
identificar inequivocamente fusoes e fragmentacoes
de instancias.

A analise de sensibilidade investiga mecanismos
potenciais de subcontagem, mas nao demonstra
isoladamente todas as suas causas.

Os resultados anteriores do Cellpose permanecem
inalterados. Nenhuma nova variante neural sera
ajustada nesta etapa.
