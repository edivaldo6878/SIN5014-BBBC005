# SIN5014 — Adendo metodológico do método C
Data: 24/11/2026. Documento complementar elaborado após o início da execução integral
C0/C1 (passo 102; log já contém 64/1200 inferências). Não constitui pré-registro.
Os parâmetros C0/C1 estavam fixados no executor versionado no commit 0fa6129 antes
da execução integral. Nenhuma mudança no executor ou nos resultados é proposta.

## Configurações
- C0: Cellpose 3.1.1.2, nuclei, diâmetro explícito 17 px, sem calibração local.
- C1: mesmo modelo, diâmetro fixo 60.52363315435602 px, mediana dos diâmetros
  circulares equivalentes (2*sqrt(area/pi)) das máscaras C1_s01 a C1_s05.
  Usa referência binária para a escala, não modifica os pesos; não é zero-shot
  estrito sem informação local. A estimativa é baseada em cinco núcleos isolados.
- 64 px: sugestão preliminar sem origem quantitativa comprovada, substituída
  pela mediana medida. Eventual C64 será exploratório, separado e restrito à
  calibração; não substituirá C1 nem será usado para selecionar resultados.
- CPU, PyTorch 2.7.1+cpu, 4 threads, interop 1, semente 0.
- channels=[0,0], normalize=True, do_3D=False, flow_threshold=0.4,
  cellprob_threshold=0.0, min_size=15, batch_size=8, resample=True, augment=False.
- Pesos SHA-256: 89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303.
- Manifesto SHA-256 (bytes): 7f8a8f2605fbcf161db426743d7dccc6f125a51b8e33f3c200c03b5bdeec2932.

## Partições e desfechos
F1/w2, 24 níveis, 25 amostras por nível. Calibração s01–s05 (120 imagens,
40 por faixa), validação s06–s25 (480, 160 por faixa). As cinco máscaras de
calibração e o piloto C10_s01, C48_s01, C91_s01 são excluídos da comparação
principal. Partição retrospectiva interna; não é validação externa independente.
As 600 imagens por configuração são processadas para análise descritiva integral.
Comparação principal pareada B vs C0 e C1 nas mesmas 480 chaves, com MAE como
desfecho primário; C0 vs C1 como comparação de escala. Reportar viés, RMSE, MAPE,
contagens exatas, sub/supercontagens, por faixa e nível. Intervalos de confiança
por bootstrap pareado, estratificado por nível (2000 reamostragens, semente 0);
diferenças de erros absolutos por Wilcoxon pareado, ajuste Holm se múltiplos
contrastes inferenciais. Propostas de análise formuladas antes da consolidação,
mas depois do início da execução; interpretar como exploratórias/retrospectivas.
A é baseline diagnóstico com acesso à máscara de referência. B* deverá ser
calibrado apenas em s01–s05, sem alterar C0/C1 com base em validação.
Dice/IoU do legado B são pré-Watershed; para comparação espacial direta
recalcular B com a união dos rótulos finais, separadamente do legado.
Referências binárias não permitem validar instâncias individualmente.
Zeros retornados por inferência válida são resultados, não dados ausentes.
Exceções técnicas e integridade inválida devem bloquear conclusão ou ser
registradas; não substituir por zero nem excluir silenciosamente.

## Recursos e rastreabilidade
Tempo model.eval com perf_counter, exclui aquecimento, leitura e métricas.
RSS antes/depois e peak_wset acumulado do processo; não é pico por imagem.
Não comparar diretamente com tracemalloc do B. Para comparação de recursos,
medir métodos em protocolo equivalente, sessões separadas por configuração,
mesmo hardware, registrando aquecimento, escopo de medição e reinícios.
Executor conserva métricas e checkpoint CSV atômico; não todas as máscaras.
Retomada com checkpoint completo testada, interrupção parcial real não testada.
Hashes do manifesto são de bytes; conversão LF/CRLF em novo checkout pode
alterá-los. Preservar a cópia original e verificar antes de execução/retomada.
O piloto de três imagens não permite generalizar desempenho por densidade.
Não modificar parâmetros à luz dos resultados de validação já parcialmente visíveis.
O e-mail Git permanece local no histórico; decidir privacidade antes de publicar.
