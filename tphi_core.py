"""tphi_core.py - T-phi gram + 39 thermodynamic indices chart.

Thermodynamics, quality control and chart layout for global radiosonde (WMO GTS) data, including
levels without a reported dew point.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection
import warnings; warnings.filterwarnings('ignore', category=RuntimeWarning)

# Colour scheme of globalweathercharts.com
TH = dict(
    page='#ffffff', panel='#f5f8fc', navy='#0b3c6f', accent='#2f7fd8',
    isotherm='#c3d4ea', zero='#5b8fd1', dry='#d9a066', moist='#5fae93', isobar='#dbe3ee', sfc='#7b8a9e',
    temp='#c2185b', dew='#00838f', parcel='#ef6c00', cape='#ffb300', cin='#7e57c2', barb='#1f2d3d',
    hodo='plasma', card='#f2f6fc', card_edge='#a9c2e3', warn='#b00020',
)

# ------------------------------------------------------------------ thermodynamics
Rd,Cp,Lv,eps,g,K0 = 287.04,1005.7,2.501e6,0.622,9.80665,273.15; kap = Rd/Cp
es = lambda t: 6.112*np.exp(17.67*t/(t+243.5))
def wsat(t,p): e=es(t); return eps*e/(p-e)
def theta(t,p): return (t+K0)*(1000/p)**kap
def tv(t,p,w): return (t+K0)*(1+w/eps)/(1+w)
def thetae(t,td,p):
    w=wsat(td,p); Tk=t+K0; Tdk=td+K0
    Tl=1/(1/(Tdk-56)+np.log(Tk/Tdk)/800)+56
    return Tk*(1000/p)**(0.2854*(1-0.28*w))*np.exp((3.376/Tl-0.00254)*1000*w*(1+0.81*w))
def lcl(t,td,p):
    Tk,Tdk=t+K0,td+K0
    Tl=1/(1/(Tdk-56)+np.log(Tk/Tdk)/800)+56
    return p*(Tl/Tk)**(1/kap),Tl-K0
def mstep(tk,p):
    r=wsat(tk-K0,p); return (Rd*tk+Lv*r)/(p*(Cp+Lv**2*r*eps/(Rd*tk**2)))
def moist(t0,pgrid):
    out=[t0]; t=t0+K0
    for i in range(1,len(pgrid)):
        p=pgrid[i-1]; dp=pgrid[i]-p
        k1=mstep(t,p);k2=mstep(t+.5*dp*k1,p+.5*dp);k3=mstep(t+.5*dp*k2,p+.5*dp);k4=mstep(t+dp*k3,p+dp)
        t+=dp*(k1+2*k2+2*k3+k4)/6; out.append(t-K0)
    return np.array(out)
def wetbulb(t,td,p):
    pl,tl=lcl(t,td,p)
    if pl>=p: return t
    return moist(tl,np.linspace(pl,p,60))[-1]
def parcel(S,p0,t0,td0,virtual=True):
    P=S['p']; m=P<=p0; pp=np.r_[p0,P[m]]
    pl,tl=lcl(t0,td0,p0); w0=wsat(td0,p0)
    tp=np.empty_like(pp); dry=pp>=pl
    tp[dry]=(t0+K0)*(pp[dry]/p0)**kap-K0
    if (~dry).any(): tp[~dry]=moist(tl,np.r_[pl,pp[~dry]])[1:]
    te=np.interp(np.log(pp),np.log(P[::-1]),S['t'][::-1])
    tde=np.interp(np.log(pp),np.log(P[::-1]),S['td'][::-1])
    if virtual:
        wp=np.where(dry,w0,wsat(tp,pp)); bp=tv(tp,pp,wp); be=tv(te,pp,wsat(tde,pp))
    else: bp=tp+K0; be=te+K0
    return pp,tp,bp-be,pl,tl
def cape_cin(pp,dT):
    dl=-np.diff(np.log(pp)); dm=.5*(dT[1:]+dT[:-1])
    pos=np.where(dm>0)[0]
    if not len(pos): return 0.,0.,np.nan,np.nan
    groups=np.split(pos,np.where(np.diff(pos)>1)[0]+1)
    areas=[Rd*np.sum(dm[gp]*dl[gp]) for gp in groups]
    lfc_i=groups[int(np.argmax(areas))][0]; el_i=groups[-1][-1]+1
    idx=np.arange(len(dm))
    return (Rd*np.sum((dm*dl)[(dm>0)&(idx>=lfc_i)]), Rd*np.sum((dm*dl)[(dm<0)&(idx<lfc_i)]),
            pp[lfc_i], pp[el_i])

QC_NOTE=['']

def quality_control(R):
    """R = columns P,T,RH,Td,Z,D,S (surface first). Removes impossible values and spikes."""
    keep=[0]
    for i in range(1,len(R)):
        if R[i,0]<R[keep[-1],0]: keep.append(i)
    R=R[keep]
    if len(R)<10: raise ValueError(f"only {len(R)} usable levels")
    # --- quality control: remove impossible temperatures and isolated spikes (partial sensor failure)
    P_,T_=R[:,0],R[:,1]
    bad=(T_>50)|(T_<-95)|((P_>=300)&(T_<-60))|((P_>=500)&(T_<-40))
    n=len(T_); w=7
    if n>max(2*w+1,300):                                      # spike test against a running median (dense soundings only:
                                                              # in a TEMP report 15 levels span hundreds of hPa)
        Tm=np.where(bad,np.nan,T_)
        pad=np.r_[np.full(w,np.nan),Tm,np.full(w,np.nan)]
        med=np.nanmedian(np.lib.stride_tricks.sliding_window_view(pad,2*w+1),axis=1)
        bad|=np.abs(T_-med)>8
    low=P_>=300; frac=bad[low].mean() if low.any() else bad.mean()
    if frac>0.2:
        raise ValueError(f"temperature sensor failure in this sounding ({frac*100:.0f}% of levels below 300 hPa "
                         f"have impossible temperatures, e.g. {np.median(T_[low&bad]) if (low&bad).any() else np.nan:.0f} °C) - not plotted")
    if bad.any():
        R=R[~bad]
    R[:,3]=np.minimum(R[:,3],R[:,1])                          # dew point never above temperature
    QC_NOTE[0]=f"{int(bad.sum())} bad temperature level(s) removed by quality control" if bad.any() else ''
    if len(R)<10: raise ValueError(f"only {len(R)} usable levels after quality control")
    return R

# ------------------------------------------------------------------ indices + figure
def make_chart(S, station, idx, when, out, source="", site="globalweathercharts.com", extra_issues=()):
    """S: dict p,t,td,rh,z,d,s (+ optional tdok: True where the dew point was reported)."""
    P,T,Td,Z,RH,D,Sp=S['p'],S['t'],S['td'],S['z'],S['rh'],S['d'],S['s']
    H=Z-Z[0]; u=-Sp*np.sin(np.radians(D)); v=-Sp*np.cos(np.radians(D))
    at=lambda arr,L: np.interp(np.log(L),np.log(P[::-1]),arr[::-1])
    ath=lambda arr,h: np.interp(h,H,arr)
    def safe(fn):
        try: return float(fn())
        except Exception: return np.nan
    p0,t0,td0=P[0],T[0],Td[0]
    sfc_note=''
    lay=(P<P[0])&(P>=P[0]-10)
    if lay.sum()>=3 and td0-np.mean(Td[lay])>5:          # surface dew point much moister than the air just above
        sfc_note=(f"surface Td {td0:.1f} °C vs {np.mean(Td[lay]):.1f} °C in lowest 10 hPa - "
                  "SBCAPE/MUCAPE may be overstated, see MLCAPE")
    pp_sb,tp_sb,b_sb,pl_sb,_=parcel(S,p0,t0,td0)
    SB,CIN,LFC,EL=cape_cin(pp_sb,b_sb)
    m=P>=p0-100; th=np.mean(theta(T[m],P[m])); wml=np.mean(wsat(Td[m],P[m]))
    tml=th*(p0/1000)**kap-K0; e=wml*p0/(eps+wml); a=np.log(e/6.112); tdml=243.5*a/(17.67-a)
    r=parcel(S,p0,tml,tdml); ML=cape_cin(r[0],r[2])[0]
    Tdx=Td
    m=P>=p0-300; i=np.argmax(thetae(T[m],Tdx[m],P[m])); pmu,tmu,tdmu=P[m][i],T[m][i],Tdx[m][i]
    r=parcel(S,pmu,tmu,tdmu); MU=cape_cin(r[0],r[2])[0]
    def lifted500(p,t,td):
        r=parcel(S,p,t,td,False); return at(T,500)-np.interp(np.log(500),np.log(r[0][::-1]),r[1][::-1])
    LI=safe(lambda: lifted500(p0,t0,td0)); SI=safe(lambda: lifted500(850,at(T,850),at(Td,850)))
    T8,T7,T5,D8,D7,D5=at(T,850),at(T,700),at(T,500),at(Td,850),at(Td,700),at(Td,500)
    KI=T8-T5+D8-(T7-D7); TT=T8+D8-2*T5
    RHmid=RH[(P<=700)&(P>=500)].mean(); RHlow=RH[P>=700].mean()
    w=wsat(Td,P); PW=-np.trapezoid(w,P*100)/g
    FZL=safe(lambda: Z[np.where(T<=0)[0][0]])
    def dcape():
        m=P>=400; te=thetae(T[m],Td[m],P[m]); i=np.argmin(te); ps=P[m][i]
        tw=wetbulb(T[m][i],Td[m][i],ps); grid=np.r_[ps,P[P>ps][::-1]]
        bb=moist(tw,grid)-at(T,grid); dl=np.diff(np.log(grid))
        return -Rd*np.sum(.5*(bb[1:]+bb[:-1])*dl), thetae(t0,td0,p0)-te.min()
    try: DCAPE,TEDEF=dcape()
    except Exception: DCAPE=TEDEF=np.nan
    VMAX=np.sqrt(DCAPE) * np.sqrt(2) if np.isfinite(DCAPE) and DCAPE>0 else np.nan
    LR06=safe(lambda:(t0-ath(T,6000))/6); LR75=(T7-T5)/(at(Z,500)-at(Z,700))*1000
    T67,D67=at(T,670),at(Td,670); LR8_67=(T8-T67)/(at(Z,670)-at(Z,850))*1000
    MWPI=SB/100+LR8_67/5+((T8-D8)-(T67-D67))/5
    DMI=LR75+(T7-D7)-(T5-D5)
    dT=np.diff(T); invb=invt=invs=np.nan; i=0
    while i<len(dT):
        if dT[i]>0:
            j=i
            while j<len(dT) and dT[j]>=0: j+=1
            if T[j]-T[i]>=0.5: invb,invt,invs=H[i],H[j],T[j]-T[i]; break
            i=j
        else: i+=1
    gr=np.diff(T)/np.maximum(np.diff(H),1e-3)*1000; GMAX=safe(lambda: np.nanmax(np.where(H[1:]<3000,gr,np.nan)))
    thv=theta(T,P)*(1+0.61*w)
    Ri=g/thv[0]*(thv-thv[0])*H/np.maximum((u-u[0])**2+(v-v[0])**2,0.1)
    BLH=safe(lambda: H[np.where((Ri>0.25)&(H>50))[0][0]])
    bl=H<=(BLH if np.isfinite(BLH) else 0)+1e-6; BLW=np.mean(Sp[bl]); VC=BLH*BLW; BLQ=np.mean(w[bl])*1000
    u6,v6=ath(u,6000),ath(v,6000); SH06=np.hypot(u6-u[0],v6-v[0])
    u3,v3=ath(u,3000),ath(v,3000); SH03=np.hypot(u3-u[0],v3-v[0])
    mm=H<=6000; um,vm=u[mm].mean(),v[mm].mean(); m5=H<=500; u5,v5=u[m5].mean(),v[m5].mean()
    sh=np.array([u6-u5,v6-v5]); sh/=max(np.linalg.norm(sh),1e-6); ur,vr=um+7.5*sh[1],vm-7.5*sh[0]
    def srh(top):
        k=H<=top; uu,vv=u[k],v[k]; return np.sum((uu[1:]-ur)*(vv[:-1]-vr)-(uu[:-1]-ur)*(vv[1:]-vr))
    SRH3,SRH1=srh(3000),srh(1000)
    BRN=safe(lambda: SB/(0.5*np.hypot(um-u5,vm-v5)**2))
    VGP=np.sqrt(SB)*SH03/3000
    LCLH=np.interp(np.log(pl_sb),np.log(P[::-1]),H[::-1])
    SCP=(MU/1000)*(SRH3/50)*(0 if SH06<10 else min(SH06,20)/20)
    STP=(SB/1500)*min(max((2000-LCLH)/1000,0),1)*(SRH1/150)*(0 if SH06<12.5 else min(SH06,30)/20)
    wmu=wsat(tdmu,pmu)*1000; SHIP=MU*wmu*LR75*(-T5)*min(max(SH06,7),27)/42e6
    if MU<1300: SHIP*=MU/1300
    if LR75<5.8: SHIP*=LR75/5.8
    if np.isfinite(FZL) and FZL<2400: SHIP*=FZL/2400
    kt=1.94384; f8,f5=at(Sp,850)*kt,at(Sp,500)*kt; d8,d5=at(D,850),at(D,500)
    SWEAT=12*max(D8,0)+20*max(TT-49,0)+2*f8+f5
    if 130<=d8<=250 and 210<=d5<=310 and d5-d8>0 and f8>=15 and f5>=15: SWEAT+=125*(np.sin(np.radians(d5-d8))+0.2)
    ix=np.arange(0,len(P),5); twb=np.array([wetbulb(T[k],Td[k],P[k]) for k in ix])
    WBZ=safe(lambda: H[ix][np.where(twb<=0)[0][0]])
    def windex():
        HM=(FZL-Z[0])/1000; QL=np.mean(w[H<=1000])*1000; pm=np.interp(FZL,Z,P); QM=wsat(at(Td,pm),pm)*1000
        GAM=t0/HM; return 5*np.sqrt(max(HM*min(QL/12,1)*(GAM**2-30+QL-2*QM),0))
    WINDEX=safe(windex)

    issues=[x for x in (QC_NOTE[0], sfc_note, *extra_issues) if x]
    if not (2<=PW<=90): issues.append(f"PW {PW:.1f} mm")
    if SB>7000 or MU>7000: issues.append(f"CAPE {max(SB,MU):.0f} J/kg")
    if not (-40<=KI<=60): issues.append(f"K Index {KI:.0f}")
    if P.min()>300: issues.append(f"ascent ends at {P.min():.0f} hPa")
    if np.any((np.abs(np.diff(T))>8)&(np.abs(np.diff(P))<20)): issues.append("temperature jump >8 °C within 20 hPa")
    fig=plt.figure(figsize=(10.5,19.5),facecolor=TH['page'])
    fig.text(0.03,0.985,'T-phi gram and Thermo-dynamic Indices Analysis',fontsize=21,weight='bold',va='top',color=TH['navy'])
    fig.text(0.03,0.968,f'{station.title()} ({idx}) | {when}',fontsize=13,va='top',color=TH['navy'])
    if issues: fig.text(0.97,0.968,'⚠ data check: '+'; '.join(issues),fontsize=9,color=TH['warn'],ha='right',va='top')
    ax=fig.add_axes([0.16,0.585,0.58,0.355]); C=35.; ax.set_facecolor(TH['panel'])
    for sp in ax.spines.values(): sp.set_color(TH['accent'])
    X=lambda t,p: t+C*np.log(1000/p)
    ax.set_yscale('log'); ax.set_ylim(1000,100); ax.set_xlim(-30,50)
    pt=[1000,900,800,700,600,500,400,300,200,100]
    ax.set_yticks(pt); ax.set_yticklabels([str(x) for x in pt]); ax.minorticks_off(); ax.tick_params(labelsize=8)
    pg=np.geomspace(1050,90,100)
    for t in range(-120,60,10): ax.plot(X(t,pg),pg,color=TH['zero'] if t==0 else TH['isotherm'],lw=1.0 if t==0 else .6)
    for th_ in range(-40,200,10): ax.plot(X((th_+K0)*(pg/1000)**kap-K0,pg),pg,color=TH['dry'],lw=.6,alpha=.6)
    mg=np.r_[1000,np.geomspace(990,100,80)]
    for t0_ in range(-10,36,5): ax.plot(X(moist(t0_,mg),mg),mg,color=TH['moist'],lw=.6,ls='--',alpha=.65)
    for p in pt: ax.axhline(p,color=TH['isobar'],lw=.6)
    ax.axhline(p0,color=TH['sfc'],lw=.8,ls='--')
    k=P>=100
    ax.plot(X(T[k],P[k]),P[k],color=TH['temp'],lw=1.6,label='Temperature'); tdok=S.get('tdok',np.ones(len(P),bool)); ax.plot(X(np.where(tdok,Td,np.nan)[k],P[k]),P[k],color=TH['dew'],lw=1.6,label='Dew point')
    kp=pp_sb>=100; tpar=tp_sb[kp]; ppar=pp_sb[kp]; ax.plot(X(tpar,ppar),ppar,color=TH['parcel'],lw=1.5,ls=(0,(5,2)),label='Surface parcel')
    te_=at(T,ppar)
    ax.fill_betweenx(ppar,X(te_,ppar),X(tpar,ppar),where=(tpar>te_)&(ppar<=LFC)&(ppar>=EL),color=TH['cape'],alpha=.38,lw=0,label='CAPE')
    ax.fill_betweenx(ppar,X(te_,ppar),X(tpar,ppar),where=(tpar<te_)&(ppar>=LFC),color=TH['cin'],alpha=.2,lw=0,label='CIN')
    ax.legend(loc='lower left',fontsize=7,frameon=True,facecolor='white',edgecolor=TH['card_edge'],framealpha=.92)
    ax.set_xlabel('degree_Celsius',fontsize=8); ax.set_ylabel('hPa',fontsize=8)
    f0=lambda x,fmt: (fmt.format(x) if np.isfinite(x) else 'n/a')
    box=(f"CAPE: {SB:.0f} J/kg\nCIN: {CIN:.0f} J/kg\nLI: {f0(LI,'{:.1f}')}\nKI: {KI:.1f}\nTTI: {TT:.1f}\n"
         f"SWEAT: {SWEAT:.0f}\nSRH: {SRH3:.0f} m²/s²\nSCP: {SCP:.1f}\nSTP: {STP:.2f}\nWBZ: {f0(WBZ,'{:.0f}')} m")
    ax.text(0.015,0.985,box,transform=ax.transAxes,fontsize=7,va='top',bbox=dict(fc='white',ec=TH['accent'],lw=.7),color=TH['navy'])
    bp=np.r_[np.arange(950,100,-50),100]; bp=bp[bp<=p0]
    ax.barbs(np.full(len(bp),50),bp,at(u,bp)*kt,at(v,bp)*kt,length=6,lw=.7,clip_on=False,color=TH['barb'])
    hx=fig.add_axes([0.505,0.84,0.18,0.10]); hk=H<=25000
    for rr in (10,20,30,40,50,60): hx.add_patch(plt.Circle((0,0),rr,fill=False,ls='--',color=TH['isotherm'],lw=.6))
    hx.axhline(0,color=TH['sfc'],lw=.5,ls='--'); hx.axvline(0,color=TH['sfc'],lw=.5,ls='--'); hx.set_facecolor(TH['panel'])
    pts=np.c_[u[hk],v[hk]].reshape(-1,1,2); seg=np.concatenate([pts[:-1],pts[1:]],axis=1)
    lc=LineCollection(seg,cmap=TH['hodo'],norm=plt.Normalize(0,25)); lc.set_array(H[hk][:-1]/1000); lc.set_linewidth(1.3)
    hx.add_collection(lc); hx.set_xlim(-60,60); hx.set_ylim(-60,60); hx.set_aspect('equal'); hx.tick_params(labelsize=6)
    hx.set_xticks([-60,-30,0,30,60]); hx.set_yticks([-60,-30,0,30,60]); hx.set_title('Hodograph (m/s)',fontsize=7,pad=2,color=TH['navy'])
    cx=fig.add_axes([0.83,0.84,0.012,0.09]); cb=fig.colorbar(lc,cax=cx); cb.set_label('Height (km)',fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.add_artist(plt.Line2D([0.03,0.97],[0.548,0.548],color=TH['accent'],lw=1.6,transform=fig.transFigure))
    fig.text(0.03,0.54,'Thermo-dynamic Indices',fontsize=16,weight='bold',va='top',color=TH['navy'])
    cards=[("Dry Mid-Levels RH (%)",RHmid),("Low-Levels RH (%)",RHlow),("Freezing Level (m)",FZL),
    ("Total Precipitable Water (mm)",PW),("Surface-Based Convective\nAvailable Potential Energy",SB),("Mixed-Layer Convective\nAvailable Potential Energy",ML),
    ("Most-Unstable Convective\nAvailable Potential Energy",MU),("Convective Inhibition (J/kg)",CIN),("Lifted Condensation Level\nHeight AGL (m)",LCLH),
    ("Downdraft Convective Available\nPotential Energy (J/kg)",DCAPE),("Maximum Downdraft Velocity\nPotential (m/s)",VMAX),("Lifted Index (°C)",LI),
    ("Showalter Index (°C)",SI),("Total Totals Index",TT),("K Index",KI),
    ("ThetaE Deficit (K)",TEDEF),("Microburst Windspeed Potential\nIndex",MWPI),("Dry Microburst Index",DMI),
    ("Wind Index, WINDEX (kt)",WINDEX),("Wet-Bulb Zero Height AGL (m)",WBZ),("SWEAT Index",SWEAT),
    ("Level of Free Convection (hPa)",LFC),("0 to 6 km Environmental Lapse\nRate (°C/km)",LR06),("700 to 500 hPa Lapse Rate\n(°C/km)",LR75),
    ("Inversion Base Height AGL (m)",invb),("Inversion Top Height AGL (m)",invt),("Inversion Strength (°C)",invs),
    ("Maximum Inversion Gradient\n(°C/km)",GMAX),("Boundary Layer Height (m)",BLH+Z[0]),("Boundary Layer Height AGL (m)",BLH),
    ("Mean Boundary Layer Wind Speed\n(m/s)",BLW),("Ventilation Coefficient (m²/s)",VC),("Boundary Layer Mixing Ratio\n(g/kg)",BLQ),
    ("Bulk Shear (0 to 6 km, m/s)",SH06),("Bulk Richardson Number",BRN),("Vorticity Generation Parameter",VGP),
    ("Storm-Relative Helicity\n0–3 km (m²/s²)",SRH3),("Significant Hail Parameter",SHIP),("Equilibrium Level (hPa)",EL)]
    globals()['LAST_CARDS']=cards    
    cw,ch,top=0.30,0.0335,0.522
    for n,(lab,val) in enumerate(cards):
        r,c=divmod(n,3); x=0.03+c*(cw+0.02); y=top-r*(ch+0.0055)-ch
        fig.add_artist(FancyBboxPatch((x,y),cw,ch,boxstyle="round,pad=0,rounding_size=0.006",transform=fig.transFigure,fc=TH['card'],ec=TH['card_edge'],lw=.8))
        fig.text(x+0.01,y+ch-0.004,lab,fontsize=8.5,va='top',linespacing=1.1)
        fig.text(x+0.01,y+0.004,(f"{float(val):.2f}" if np.isfinite(val) else "n/a"),fontsize=11.5,weight='bold',va='bottom',color=TH['navy'])
    fig.text(0.03,0.012,source or f'Source: computed from the radiosonde ascent of {station.title()} ({idx}), {when}',fontsize=9)
    if site: fig.text(0.97,0.012,site,fontsize=10,weight='bold',color=TH['accent'],ha='right')
    fig.savefig(out,dpi=130); plt.close(fig)
    return dict(SBCAPE=SB,MUCAPE=MU,CIN=CIN,LI=LI,KI=KI,PW=PW,suspect='; '.join(issues))
