#!/usr/bin/env python3
# SIN5014 - ETAPA FINAL - PASSO 237
"""Auditoria de evidencias, preservacao de bytes e fechamento Git. Sem inferencia.
Execute com a coleta original e o pacote de figuras fornecidos ao lado do script.
"""
import argparse, csv, datetime as dt, hashlib, io, json, math, os, re, shutil, subprocess, sys, tempfile, zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

SNAPSHOT='SIN5014_FECHAMENTO_COMPLETO_20261125_113330_184069.zip'
SNAP_SHA='182404acd3104b47809cd2650e34e23b12c1bb5596691899330395a9fdda2ebd'
EVIDENCE='entrega_final_20261125_104334.zip'
EV_SHA='d63f3e19bcf4cb1ce28f6ecdac8780731bc9041d7de4029935b98dea6662a06b'
FINAL=EVIDENCE[:-4]; CAL='watershed_bstar_calibracao_20261125_095745'
TAG='sin5014-experimentos-v1'
LEVELS=(1,5,10,14,18,23,27,31,35,40,44,48,53,57,61,66,70,74,78,83,87,91,96,100)
METHODS=('A','B','Bstar','C0','C1','Cstar')
GRID={f'B{i*3+j+1}':(a,b) for i,a in enumerate((.35,.50,.65)) for j,b in enumerate((.20,.30,.40))}

def need(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(b): return hashlib.sha256(b).hexdigest()
def jb(obj): return (json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8')
def csvrows(data): return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
def near(x,y): return math.isclose(float(x),float(y),rel_tol=0,abs_tol=1e-9)
def group(n): return 'Baixa' if n<=31 else 'Media' if n<=66 else 'Alta'
def pathcheck(name):
    p=PurePosixPath(name)
    need(name==p.as_posix() and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and '\n' not in name and '\r' not in name,'Unsafe archive path: '+name)

def unpack(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        need(z.testzip() is None,'ZIP CRC failure')
        names=[n for n in z.namelist() if not n.endswith('/')]
        need(len(set(names))==len(names),'Duplicate ZIP entry')
        for n in names: pathcheck(n)
        return {n:z.read(n) for n in names}

def check_sums(entries,key,all_entries=False):
    prefix=key.rsplit('/',1)[0]+'/' if '/' in key else ''
    checked=set()
    for line in entries[key].decode('utf-8-sig').splitlines():
        if not line.strip():continue
        h,n=line.split(maxsplit=1); n=prefix+n.strip().lstrip('*');pathcheck(n)
        need(n not in checked and n in entries and sha(entries[n])==h.lower(),'SHA256 mismatch: '+n);checked.add(n)
    if all_entries: need(checked==set(entries)-{key},'Incomplete checksum inventory: '+key)
    return len(checked)

def keyed(rows,fields):
    d={}
    for r in rows:
        k=tuple(int(r[x]) if x in ('n','amostra','contagem_esperada') else r[x] for x in fields)
        need(k not in d,'Duplicate key: '+str(k));d[k]=r
    return d

def audit(ev):
    import numpy as np
    k=unpack(ev); check_sums(k,FINAL+'/SHA256SUMS.txt')
    sel=json.loads(k[CAL+'/selecao_bstar.json']); cp=json.loads(k[CAL+'/protocolo.json']); vp=json.loads(k[FINAL+'/protocolo.json'])
    need(sel['selected']=='B1' and (sel['alpha'],sel['beta'])==GRID['B1'],'Selection mismatch')
    need({x:tuple(y) for x,y in cp['grid'].items()}==GRID,'Grid mismatch')
    for n,h in [(CAL+'/por_imagem_bstar.csv',sel['calibration_csv_sha256']),(CAL+'/protocolo.json',sel['protocol_sha256']),(CAL+'/selecao_bstar.json',vp['selection_sha256']),('inputs/scripts/executar_watershed_bstar_v1.py',sel['executor_sha256']),('inputs/docs/protocolo_watershed_bstar.md',cp['methodological_protocol_sha256']),('inputs/scripts/fechar_experimentos_sin5014.py',vp['script_sha256']),('inputs/manifestos/manifest_f1_w2.csv',vp['manifest_sha256'])]:need(sha(k[n])==h,'Protocol hash mismatch: '+n)
    need(vp['selected']==sel and vp['partition']=='validacao' and vp['seed']==0 and vp['bootstrap']==2000,'Validation protocol mismatch')
    for n,h in vp['source_hashes'].items():
        f,d=n.split(':');need(sha(k['inputs/'+d+'/'+f])==h,'Input hash mismatch: '+n)
    baseline=keyed(csvrows(k['inputs/reproducao_v04_20261124_201858/por_imagem.csv']),('contagem_esperada','amostra'))
    keys={(n,s) for n in LEVELS for s in range(1,26)}
    need(set(baseline)==keys,'Baseline incomplete')
    parts={}
    for part,fn,tags,samples in [('calibracao',CAL+'/por_imagem_bstar.csv',set(GRID),range(1,6)),('validacao',FINAL+'/por_imagem_bstar.csv',{'B1','B5'},range(6,26))]:
        rows=keyed(csvrows(k[fn]),('config','contagem_esperada','amostra'))
        need(set(rows)=={(tag,n,s) for tag in tags for n in LEVELS for s in samples},'Incomplete '+part)
        for (tag,n,s),r in rows.items():
            count=int(r['ws_n']); err=int(r['ws_erro'])
            need(count>=0 and count==int(r['marcadores']) and err==count-n and r['particao']==part and r['grupo']==group(n),'Bad Watershed row')
            need((float(r['alpha']),float(r['beta']))==GRID[tag],'Parameter mismatch')
            for p in ('pre_','final_'):
                tp,fp,fn=(int(r[p+x]) for x in ('tp','fp','fn')); need(min(tp,fp,fn)>=0,'Negative pixel count')
                need(near(r[p+'dice'],2*tp/max(1,2*tp+fp+fn)) and near(r[p+'iou'],tp/max(1,tp+fp+fn)) and near(r[p+'precision'],tp/max(1,tp+fp)) and near(r[p+'recall'],tp/max(1,tp+fn)),'Binary metric mismatch')
            need(int(r['pixels_final'])<=int(r['pixels_pre']) and int(r['pixels_sem_rotulo'])==int(r['pixels_pre'])-int(r['pixels_final']),'Coverage mismatch')
            if tag=='B5':
                old=baseline[n,s];need(count==int(old['ws_n']) and err==int(old['ws_erro']) and near(r['pre_dice'],old['ws_dice']) and near(r['pre_iou'],old['ws_iou']),'B5 baseline mismatch: '+str((n,s)))
        parts[part]=rows
    rank={t:sum(abs(int(r['ws_erro'])) for (tag,n,s),r in parts['calibracao'].items() if tag==t) for t in GRID}
    winner=min(rank,key=lambda t:(rank[t],0 if t=='B5' else int(t[1:])))
    need(winner=='B1' and rank[winner]==1634 and rank['B5']==2010 and near(sel['mae'],1634/120),'Calibration selection mismatch')
    d=keyed(csvrows(k[FINAL+'/resultados_seis_metodos.csv']),('metodo','n','amostra'))
    need(set(d)=={(m,n,s) for m in METHODS for n,s in keys},'Consolidated keys mismatch')
    neural={}
    for f in ['inputs/cellpose_completo_20261124_214400/por_imagem_cellpose.csv','inputs/cellpose_cstar_calibracao_20261125_081650/por_imagem_cellpose.csv','inputs/cellpose_cstar_validacao_20261125_083155/por_imagem_cellpose.csv']:
        for r in csvrows(k[f]):
            kk=({'C0_fixed17':'C0','C1_calibrated':'C1','Cstar_norm':'Cstar'}[r['config']],int(r['contagem_esperada']),int(r['amostra']));need(kk not in neural,'Neural duplicate');neural[kk]=r
    need(set(neural)=={(m,n,s) for m in ('C0','C1','Cstar') for n,s in keys},'Neural inputs incomplete')
    for (m,n,s),r in d.items():
        e=int(r['erro']); p='calibracao' if s<=5 else 'validacao'; c=int(r['contagem'])
        need(e==c-n and abs(e)==int(r['ae']) and r['particao']==p and r['grupo']==group(n),'Consolidated error mismatch')
        if m in ('B','Bstar'):
            q=parts[p][('B5' if m=='B' else 'B1',n,s)];need(c==int(q['ws_n']) and near(r['dice'],q['final_dice']) and near(r['iou'],q['final_iou']),'Watershed consolidation mismatch')
        elif m=='A': need(c==int(baseline[n,s]['n8_ref']) and not r['dice'] and not r['iou'],'A control mismatch')
        else:
            q=neural[m,n,s];need(c==int(q['cp_n']) and near(r['dice'],q['cp_dice']) and near(r['iou'],q['cp_iou']),'Neural consolidation mismatch')
    summ=keyed(csvrows(k[FINAL+'/resumo_seis_metodos.csv']),('particao','grupo','metodo'))
    need(len(summ)==72,'Summary incomplete')
    for (p,g,m),r in summ.items():
        rr=[x for (mm,n,s),x in d.items() if mm==m and (p=='Todas' or x['particao']==p) and (g=='Todas' or x['grupo']==g)]; e=[int(x['erro']) for x in rr]; length=len(e)
        need(length==int(r['n_imagens']) and near(r['mae'],sum(map(abs,e))/length) and near(r['vies'],sum(e)/length) and near(r['rmse'],math.sqrt(sum(x*x for x in e)/length)) and near(r['mape'],100*sum(abs(int(x['erro']))/int(x['n']) for x in rr)/length),'Summary mismatch')
        for field,test in [('exatas',lambda x:x==0),('subcontagens',lambda x:x<0),('supercontagens',lambda x:x>0)]:need(sum(test(x) for x in e)==int(r[field]),'Summary count mismatch')
        need(sum(int(x['contagem'])==0 for x in rr)==int(r['zeros']),'Zero mismatch')
        for field,col in [('dice_final','dice'),('iou_final','iou')]:
            need((not r[field]) if m=='A' else near(r[field],sum(float(x[col]) for x in rr)/length),'Summary overlap mismatch')
    ci=keyed(csvrows(k[FINAL+'/comparacoes_pareadas.csv']),('contraste','grupo'))
    contrasts=['B-A','Bstar-B','C0-B','C1-B','C1-C0','Cstar-C1','Cstar-B','C1-Bstar']
    need(set(ci)=={(c,g) for c in contrasts for g in ('Baixa','Media','Alta','Todas')},'Contrast keys mismatch')
    for (c,g),r in ci.items():
        m1,m2=c.split('-'); rng=np.random.default_rng(0); acc=np.zeros(2000); all_e=[]
        for n in LEVELS:
            if g!='Todas' and group(n)!=g:continue
            v=np.array([int(d[m1,n,s]['ae'])-int(d[m2,n,s]['ae']) for s in range(6,26)])
            all_e.extend(v.tolist());acc+=v[rng.integers(0,len(v),(2000,len(v)))].sum(axis=1)
        a=np.array(all_e);lo,hi=np.quantile(acc/len(a),[.025,.975]);need(int(r['n_imagens'])==len(a) and near(r['delta_mae'],a.mean()) and near(r['ic95_inf'],lo) and near(r['ic95_sup'],hi) and int(r['melhor'])==int((a<0).sum()) and int(r['empate'])==int((a==0).sum()) and int(r['pior'])==int((a>0).sum()),'Bootstrap or contrast mismatch: '+str((c,g)))
    return {'status':'APROVADO','scope':'Auditoria de arquivos e resultados existentes; sem nova inferencia; validacao interna retrospectiva','calibration_rows':1080,'validation_rows':960,'b5_controls':600,'consolidated_rows':3600,'summary_rows':72,'bootstrap_intervals_recalculated':32,'bootstrap_seed':0,'bootstrap_replicates':2000,'calibration_absolute_error_sums':rank,'selected':'B1','alpha':.35,'beta':.2,'validation_mae':{m:float(summ['validacao','Todas',m]['mae']) for m in METHODS},'validation_bstar_minus_b':ci['Bstar-B','Todas'],'evidence_sha256':sha(ev)}

def zipout(path,entries):
    entries=dict(entries);entries['SHA256SUMS.txt']=''.join(f'{sha(v)}  {k}\n' for k,v in sorted(entries.items())).encode('utf-8')
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for k,v in sorted(entries.items()):z.writestr(k,v)
    check_sums(unpack(path.read_bytes()),'SHA256SUMS.txt',True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--backup-dir',type=Path,required=True);ap.add_argument('--remote');ap.add_argument('--publish',action='store_true');ap.add_argument('--out-dir',type=Path,default=Path.home()/'Downloads');a=ap.parse_args()
    root=a.repo.resolve(); here=Path(__file__).resolve().parent; stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    need(not out.is_relative_to(root),'Output must be outside Git repo')
    report={'local_status':'PENDENTE','publication_status':'NAO_SOLICITADA','tag':TAG,'errors':[]};log=[];payload={};targets={}
    def note(s):log.append(str(s));print(s,flush=True)
    def git(*args,allowed=(0,),data=None,timeout=120):
        p=subprocess.run(['git','-C',str(root),*args],input=data,capture_output=True,timeout=timeout)
        # Never include remote/auth output in exported logs. Error category only.
        need(p.returncode in allowed,'Git '+args[0]+' failed: exit '+str(p.returncode))
        return p
    def txt(*args):return git(*args).stdout.decode('utf-8','replace').strip()
    def local(rel):
        pathcheck(rel);p=root/rel;need(p.resolve().is_relative_to(root) and not p.is_symlink(),'Unsafe local path: '+rel);return p
    def materialize(rel,b,allowed_old=None):
        p=local(rel)
        if p.exists():need(p.is_file(),'Not a file: '+rel);old=p.read_bytes();need(old==b or (allowed_old is not None and old==allowed_old),'Unreviewed local change: '+rel)
        else:old=None
        if old!=b:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        targets[rel]=b
    try:
        need(Path(txt('rev-parse','--show-toplevel')).resolve()==root,'Use repository root')
        report['head_before']=txt('rev-parse','HEAD');need(txt('branch','--show-current')=='main','Expected branch main')
        snap=(here/SNAPSHOT).read_bytes();need(sha(snap)==SNAP_SHA,'Original collection SHA mismatch');outer=unpack(snap);check_sums(outer,'SHA256SUMS_COLETA.txt',True)
        ev=outer['SIN5014_Evidencias_Originais.zip'];need(sha(ev)==EV_SHA,'Evidence SHA mismatch');need(local('results/'+EVIDENCE).read_bytes()==ev,'Local original evidence differs')
        payload['SIN5014_Evidencias_Originais.zip']=ev
        ar=audit(ev);payload['AUDITORIA_BSTAR.json']=jb(ar);note('B* APROVADO: 600 controles B5; 3600 linhas; 32 ICs recalculados.')
        code=unpack(outer['SIN5014_Codigo_Fonte.zip']);res=unpack(outer['SIN5014_Resultados_e_Figuras.zip'])
        for item in (code,res):check_sums(item,'SHA256SUMS_COLETA.txt',True)
        oldread=code['README.md']
        for rel,b in {**code,**res}.items():
            if rel in ('.gitattributes','README.md','COLETA_IDENTIDADE.json','SHA256SUMS_COLETA.txt'):continue
            need(local(rel).is_file() and local(rel).read_bytes()==b,'Local file differs from audited collection: '+rel)
            targets[rel]=b
        additions=unpack((here/'artefatos_reproducibilidade.zip').read_bytes());check_sums(additions,'SHA256SUMS.txt',True);additions.pop('SHA256SUMS.txt')
        additions['docs/README_historico.md']=oldread;additions['docs/auditoria_bstar_final.json']=jb(ar)
        additions['scripts/fechar_git_bstar.py']=Path(__file__).read_bytes()
        scope=set(targets)|set(additions)|{'.gitattributes'}
        staged=set(git('diff','--cached','--name-only','-z').stdout.decode('utf-8').split('\0'))-{''}
        need(staged<=scope,'Unrelated staged files: '+', '.join(sorted(staged-scope)))
        # Refuse merging, rebasing, or unmerged paths.
        need(not git('ls-files','-u').stdout,'Unmerged index')
        for state in ('MERGE_HEAD','CHERRY_PICK_HEAD','REVERT_HEAD','rebase-merge','rebase-apply'):
            p=Path(txt('rev-parse','--git-path',state));p=p if p.is_absolute() else root/p;need(not p.exists(),'Git operation in progress: '+state)
        for rel,b in additions.items():materialize(rel,b,oldread if rel=='README.md' else None)
        attrs=local('.gitattributes');old=attrs.read_bytes() if attrs.exists() else b''
        rules=[('/'+rel+' -text -filter -working-tree-encoding -ident').encode('ascii') for rel in sorted(set(targets)|{'.gitattributes'})]
        missing=[r for r in rules if r not in set(old.splitlines())]
        new=old+(b'\n' if old and not old.endswith(b'\n') else b'')+b'# SIN5014 - exact scientific artifact bytes\n'+b'\n'.join(missing)+b'\n' if missing else old
        if old!=new:attrs.write_bytes(new)
        targets['.gitattributes']=new
        # Direct, byte-exact index insertion. Preserve existing regular-file modes.
        modes={}
        for raw in git('ls-files','--stage','-z').stdout.split(b'\0'):
            if not raw:continue
            meta,rel=raw.split(b'\t',1);mode,_,stage=meta.decode().split();need(stage=='0','Unmerged path');modes[rel.decode('utf-8')]=mode
        for rel,b in sorted(targets.items()):
            mode=modes.get(rel,'100644');need(mode in ('100644','100755'),'Unexpected file mode: '+rel)
            oid=git('hash-object','-w','--no-filters','--stdin',data=b).stdout.decode().strip()
            git('update-index','--add','--cacheinfo',f'{mode},{oid},{rel}')
            need(git('show',':'+rel).stdout==b,'Index bytes mismatch: '+rel)
        staged=set(git('diff','--cached','--name-only','-z').stdout.decode('utf-8').split('\0'))-{''}
        need(staged<=set(targets),'Unexpected staged paths')
        if staged:git('commit','-m','chore: seal audited Bstar evidence and reproducibility artifacts');note('Commit criado.')
        head=txt('rev-parse','HEAD');report['head']=head
        for rel,b in targets.items():need(git('show',head+':'+rel).stdout==b,'HEAD bytes mismatch: '+rel)
        git('fsck','--full',timeout=240)
        t=git('rev-parse','--verify','refs/tags/'+TAG+'^{}',allowed=(0,128))
        if t.returncode==0:need(t.stdout.decode().strip()==head,'Existing tag points to another commit; not modified')
        else:git('tag','-a',TAG,'-m','SIN5014 audited experimental snapshot',head)
        backup=a.backup_dir.resolve();need(not backup.is_relative_to(root),'Backup must be outside repository');backup.mkdir(parents=True,exist_ok=True)
        bpath=backup/('SIN5014_'+stamp+'.bundle');git('bundle','create',str(bpath),'--all',timeout=240);git('bundle','verify',str(bpath))
        ec=backup/EVIDENCE
        if ec.exists():need(ec.read_bytes()==ev,'Existing evidence backup differs')
        else:ec.write_bytes(ev)
        need(ec.read_bytes()==ev,'Evidence backup verification failed')
        # Verify a real restored checkout, independent of the working tree and index.
        with tempfile.TemporaryDirectory(prefix='sin5014_restore_') as tmp:
            restored=Path(tmp)/'clone'
            git('-c','core.autocrlf=true','-c','core.longpaths=true','clone','--quiet','--branch',TAG,str(bpath),str(restored),timeout=240)
            for rel,b in targets.items():
                need((restored/rel).is_file() and (restored/rel).read_bytes()==b,'Restored checkout mismatch: '+rel)
        report.update(local_status='APROVADO',verified_files=len(targets),restored_checkout_verified=True,bundle_sha256=sha(bpath.read_bytes()),bundle_file=bpath.name)
        payload[bpath.name]=bpath.read_bytes();payload['GIT_LOG.txt']=git('log','-20','--oneline').stdout;payload['GIT_STATUS.txt']=git('status','--short').stdout
        payload['INVENTARIO_GIT_SHA256.txt']=''.join(f'{sha(b)}  {rel}\n' for rel,b in sorted(targets.items())).encode()
        # Code/figures export is generated from the verified Git snapshot, not arbitrary local files.
        with tempfile.TemporaryDirectory() as tmp:
            zp=Path(tmp)/'SIN5014_Codigo_Fonte_e_Figuras.zip'
            zipout(zp,{r:b for r,b in targets.items() if not r.startswith('results/')})
            payload[zp.name]=zp.read_bytes()
        note('Git local APROVADO; tag '+TAG+'; backup verificado.')
        # Publication is opt-in; no force push and no credentials in URLs or reports.
        if a.publish:
            report['publication_status']='PENDENTE'
            need(a.remote and re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?',a.remote),'Provide actual HTTPS GitHub URL, without credentials')
            remotes=txt('remote').splitlines();need(not remotes or remotes==['origin'],'Unexpected remotes; review manually')
            if not remotes:git('remote','add','origin',a.remote)
            else:need(txt('remote','get-url','origin').removesuffix('.git')==a.remote.removesuffix('.git'),'origin differs; not modified')
            report['remote_url']=a.remote.removesuffix('.git');git('push','-u','origin','main',timeout=300);git('push','origin','refs/tags/'+TAG,timeout=300)
            remote=txt('ls-remote','origin','refs/heads/main','refs/tags/'+TAG+'^{}','refs/tags/'+TAG)
            refs=dict((l.split()[1],l.split()[0]) for l in remote.splitlines() if l.strip());need(refs.get('refs/heads/main')==head and refs.get('refs/tags/'+TAG+'^{}')==head,'Remote ref mismatch')
            report.update(publication_status='APROVADO_ACESSO_AUTENTICADO',remote_head=head,remote_tag=TAG,public_access_verified=False)
            note('Publicacao conferida: branch main e tag remotas correspondem ao commit.')
        else:report['publication_status']='PENDENTE_URL_AUTORIZACAO'
    except Exception as ex:
        report['errors'].append(type(ex).__name__+': '+str(ex));note('PENDENCIA: '+report['errors'][-1])
    report['no_new_segmentation']=True
    payload['RELATORIO_FECHAMENTO.json']=jb(report);payload['RELATORIO_OPERACIONAL.txt']=('\n'.join(log)+'\n').encode('utf-8');payload['fechar_git_bstar.py']=Path(__file__).read_bytes()
    target=out/('SIN5014_GIT_BSTAR_FINAL_'+stamp+'.zip');zipout(target,payload);print('ARQUIVO PARA ENVIO:',target,flush=True)
    return 0 if report['local_status']=='APROVADO' and not report['errors'] else 1

if __name__=='__main__':raise SystemExit(main())
