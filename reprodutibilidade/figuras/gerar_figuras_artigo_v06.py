#!/usr/bin/env python3
"""Recreate monochrome PT/EN figures from archived experimental CSVs and images; no segmentation."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ap=argparse.ArgumentParser()
ap.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parent/'inputs')
ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'figuras_regeneradas')
a=ap.parse_args(); inputs=a.inputs; figs=a.out; figs.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(inputs/'resumo_seis_metodos.csv'); d=pd.read_csv(inputs/'resultados_seis_metodos.csv'); ci=pd.read_csv(inputs/'comparacoes_pareadas.csv'); rank=pd.read_csv(inputs/'ranking_calibracao.csv')
for n in [10,48,91]:
    Image.open(inputs/f'exemplo_C{n}_s06.png').convert('L').save(figs/f'exemplo_C{n}.png')
def fmt(v,lang,n=4):
 if pd.isna(v): return '--'
 x=f'{v:.{n}f}'; return x if lang=='EN' else x.replace('.',',')
methods=['A','B','Bstar','C0','C1','Cstar']; names=['A','B','B*','C0','C1','C*']; grp=['Baixa','Media','Alta','Todas']
# Monochrome plots with unique markers and line styles. Localized axes.
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black'})
for lang in ['PT','EN']:
 en=lang=='EN'; label=lambda a,b:b if en else a
 q=s[(s.particao=='validacao')&(s.grupo=='Todas')].set_index('metodo').loc[methods]
 fig,ax=plt.subplots(figsize=(5.4,2.65)); ax.bar(names,q.mae,color=['.75','.55','.12','.85','.65','.4'],edgecolor='black',linewidth=.6); ax.set_ylabel(label('MAE (núcleos/imagem)','MAE (nuclei/image)')); ax.set_ylim(0,43)
 for i,v in enumerate(q.mae): ax.text(i,v+.65,fmt(v,lang,2),ha='center',fontsize=8)
 fig.tight_layout(); fig.savefig(figs/f'mae_{lang}.pdf'); plt.close(fig)
 fig,ax=plt.subplots(figsize=(4.5,3.0)); mat=rank.set_index('config').loc[[f'B{i}' for i in range(1,10)]].mae.to_numpy().reshape(3,3); ax.imshow(mat,cmap='Greys',vmin=10,vmax=22); ax.set_xticks(range(3),[fmt(x,lang,2) for x in [.2,.3,.4]]); ax.set_yticks(range(3),[fmt(x,lang,2) for x in [.35,.5,.65]]); ax.set_xlabel(r'$\beta$'); ax.set_ylabel(r'$\alpha$');
 for i in range(3):
  for j in range(3): ax.text(j,i, f'B{i*3+j+1}\n'+fmt(mat[i,j],lang,2),ha='center',va='center',color='white' if mat[i,j]>17.5 else 'black',fontweight='bold' if (i,j)==(0,0) else 'normal')
 fig.tight_layout(); fig.savefig(figs/f'grade_{lang}.pdf'); plt.close(fig)
 # Resolve actual column names from exported CSV.
 count_col=next(c for c in ['contagem','estimativa','detectada','contagem_detectada','pred','pred_n','n_pred'] if c in d.columns) if any(c in d.columns for c in ['contagem','estimativa','detectada','contagem_detectada','pred','pred_n','n_pred']) else None
 n_col=next(c for c in ['n','contagem_esperada'] if c in d.columns)
 if count_col is None: count_col='__count'; d[count_col]=d[n_col]+d[next(c for c in ['erro','erro_assinado'] if c in d.columns)]
 fig,ax=plt.subplots(figsize=(6.6,3.0)); val=d[d.particao=='validacao']; xx=sorted(val[n_col].unique()); ax.plot(xx,xx,ls='--',color='.45',label=label('Ideal','Ideal'),lw=1.2)
 for j,(m,nm) in enumerate(zip(methods,names)):
  y=val[val.metodo==m].groupby(n_col)[count_col].mean(); ax.plot(y.index,y,marker=['x','s','o','^','D','v'][j],markevery=2,ms=3.5,ls=[':', '--','-','-.',':','--'][j],color='black' if m=='Bstar' else str(.18+j*.09),label=nm,lw=1.6 if m=='Bstar' else .9)
 ax.set(xlabel=label('Contagem esperada','Expected count'),ylabel=label('Contagem média detectada','Mean detected count')); ax.legend(ncol=4,fontsize=8,frameon=False); fig.tight_layout(); fig.savefig(figs/f'contagens_{lang}.pdf'); plt.close(fig)
 c=ci[ci.contraste=='Bstar-B'].set_index('grupo').loc[grp]; fig,ax=plt.subplots(figsize=(5.2,2.5)); yy=np.arange(4); ax.errorbar(c.delta_mae,yy,xerr=[c.delta_mae-c.ic95_inf,c.ic95_sup-c.delta_mae],fmt='o',color='black',capsize=3); ax.axvline(0,color='.55',ls='--'); ax.set_yticks(yy,label(['Baixa','Média','Alta','Global'],['Low','Medium','High','Overall'])); ax.invert_yaxis(); ax.set_xlabel(label('Diferença de MAE (B* − B)','MAE difference (B* − B)')); fig.tight_layout(); fig.savefig(figs/f'ic_{lang}.pdf'); plt.close(fig)
 diag=pd.read_csv(inputs/'percentis_por_nivel.csv')
 fig,ax=plt.subplots(figsize=(5.2,2.6)); ax.plot(diag.n,diag.p99,'o-',color='black',ms=3,label='P99'); ax.plot(diag.n,diag.p99_9,'s--',color='.45',ms=3,label=label('P99,9','P99.9')); ax.set(xlabel=label('Contagem esperada','Expected count'),ylabel=label('Intensidade (percentil)','Intensity (percentile)')); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figs/f'percentis_{lang}.pdf'); plt.close(fig)
# Use black labels on a light gray grid.
for lang in ['PT','EN']:
 fig,ax=plt.subplots(figsize=(4.4,2.8)); mat=rank.set_index('config').loc[[f'B{i}' for i in range(1,10)]].mae.to_numpy().reshape(3,3); ax.imshow(mat,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('lightgray',['#ffffff','#aaaaaa']),vmin=13,vmax=21); ax.set_xticks(range(3),[fmt(x,lang,2) for x in [.2,.3,.4]]); ax.set_yticks(range(3),[fmt(x,lang,2) for x in [.35,.5,.65]]); ax.set_xlabel(r'$\beta$'); ax.set_ylabel(r'$\alpha$');
 for i in range(3):
  for j in range(3): ax.text(j,i, f'B{i*3+j+1}\n'+fmt(mat[i,j],lang,2),ha='center',va='center',color='black',fontweight='bold' if (i,j)==(0,0) else 'normal')
 ax.add_patch(plt.Rectangle((-.49,-.49),.98,.98,fill=False,lw=2,ec='black')); fig.tight_layout(); fig.savefig(figs/f'grade_{lang}.pdf'); plt.close(fig)

print("Figuras:",figs)
