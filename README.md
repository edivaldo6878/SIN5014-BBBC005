# SIN5014 - BBBC005

Universidade de Sao Paulo - EACH/USP
Disciplina: SIN5014 - Fundamentos de Processamento Grafico
Autor: Edivaldo Vicente dos Santos

## Objetivo

Avaliacao da acuracia na identificacao e contagem automatica de nucleos em imagens sinteticas de microscopia.

## Metodos

A - Componentes conectados N8 sobre a mascara de referencia.
B - Pipeline classico: mediana, Otsu, transformada de distancia, maximos locais e Watershed.
B* - Analise de sensibilidade dos marcadores (pendente).
C - Cellpose pre-treinado zero-shot (pendente).

## Dados

BBBC005 - Broad Bioimage Benchmark Collection.
Recorte: 600 imagens, foco F1, canal w2.
24 niveis de contagem, com 25 amostras previstas por nivel.

## Estado experimental

Pareamento das 600 imagens validado.
Reproducao preliminar de tres imagens aprovada.
Reproducao integral pendente.

## Reprodutibilidade

Codigo historico preservado em legacy/.
Ambiente computacional registrado em environment/.
Dados originais mantidos fora do Git.

## Limitacoes

A referencia binaria nao individualiza cada nucleo.
Dice e IoU binarios nao comprovam segmentacao correta por instancia.
O metodo A possui acesso privilegiado a mascara de referencia.
