"""
XDSM diagrams: the dispatch LP and the monolithic NLP.

NOTATION
  The diagrams follow the Extended Design Structure Matrix convention of Lambe and Martins (2012), cited in the thesis as [XDSM]:

      A. B. Lambe and J. R. R. A. Martins, "Extensions to the Design Structure Matrix for the Description of Multidisciplinary Design, Analysis, and
      Optimization Processes", Structural and Multidisciplinary Optimization, 46:273-284, 2012. doi:10.1007/s00158-012-0763-y
      https://websites.umich.edu/~mdolaboratory/pdf/Lambe2012a.pdf

  Overview and further examples: https://mdolab.engin.umich.edu/wiki/xdsm-overview

  Reading convention, from that source. Components sit on the diagonal; parallelograms off the diagonal are the data passed between them. Data
  travels along the rows and columns, so a component sends data vertically and receives it horizontally. The number before a colon is the process step, and
  the thin black line is the process flow. "0, 2 -> 1" on the optimizer means it starts at step 0, and returns to step 1 from step 2 until convergence.

  These are the specific variants for this thesis, not the general forms of Lambe and Martins Figures 7 and 8.

Grey data nodes carry a Scheme-A title (what the slice IS) + the variables as typeset subscripts. Wind and price enter above the objective column; the design
parameters E, P enter above the optimizer; the external output is the optimal dispatch.

  LP  diagonal : LP optimizer, Objective, Constraints
  NLP diagonal : NLP optimizer, Objective, Constraints, Gradient
                 + Rainflow+Phi analysis block feeding Objective and Gradient
                 flag on the Gradient block (that is what goes discontinuous)

Data nodes and their titles:
  Objective   reads  Power dispatch          p^s_t, p^c_t
  Constraints read   Power + energy state    p^s_t, p^c_t, e_t
  Gradient    reads  Energy trajectory       e_t

LAYOUT NOTE (2026 revision)
  Geometry is no longer hard-coded. Each figure is built at a font scale FS. Box widths are computed as (measured text width at that scale) + a FIXED
  padding, and the column and row pitches are computed as (box widths) + a FIXED clearance. Because the padding and clearance do not scale, raising FS
  makes the text grow faster than the figure, which improves legibility once the figure is scaled to fit a page. FS = 1.0 reproduces the previous layout.

  Run the module with --sweep to print the on-page text size that each FS value produces at a given LaTeX text width, then set FS_NLP / FS_LP accordingly.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase.pdfmetrics import stringWidth

# -- Output ------------------------------------------------------------------
# "png", "pdf" or "both". The tight crop measures a rendered raster either way,
# so a PNG is always produced internally; this only selects what is kept.
OUTPUT = "png"

# -- Palette ----------------------------------------------------------------
OPT=HexColor("#0076C2"); ANLY=HexColor("#7B6FA8"); DEG=HexColor("#A6543A")
GRAD=HexColor("#3F7F5F"); DATA=HexColor("#6E7B8B"); DLINE=HexColor("#8C959E")
PLINE=HexColor("#1A1A1A"); TAG=HexColor("#E9EEF2"); FLAG=HexColor("#A32D2D")
GREY=HexColor("#5A6472"); FONT="Helvetica-Bold"

# -- FONT SCALE (the knob) --------------------------------------------------
# 1.00 reproduces the previous layout. Larger values make the text bigger
# relative to the figure. Set per figure; see the --sweep output.
FS_LP  = 1.00
FS_NLP = 1.60

# Nominal font sizes at FS = 1.0
S_BODY  = 10.5   # component box labels
S_TAG   =  9.0   # "1:" process tag
S_LOOP  =  9.0   # "0, 2 -> 1" loop tag
S_TITLE =  8.4   # grey slice title
S_VARS  = 10.0   # grey slice variables
S_IO    =  9.5   # external node, single line / variables line
S_IO_T  =  8.8   # external node title when a variables line follows
S_EDGE  =  8.5   # "2: f" edge labels
S_FLAG  =  8.0   # discontinuity annotation

# -- FIXED geometry (does NOT scale with FS; this is where the gain comes from)
PAD_BOX   = 26   # horizontal padding inside a box, total
MIN_BOX   = 60   # floor on the text-driven part of a box width
CLEAR_H   = 30   # clear space between neighbouring boxes on the top row
CLEAR_V   = 36   # clear space between rows
GAP_OUT   = 32   # optimizer to external output node
GAP_TOP   = 32   # external input node to the box below it
SKEW      = 12   # parallelogram skew, must match _para
LW_DATA   = 3.0  # data connector line width
LW_PROC   = 1.1  # process line width

# Connector endpoints sit exactly on the box edge. A positive inset here
# shortens every line at BOTH ends, so any gap below 2*STUB disappears.
STUB = 0

# Module-level active scale, set by build_lp / build_nlp before drawing.
FS = 1.0

# -- Text measurement and drawing -------------------------------------------
def _rich_tokens(s):
    """Split a string on ^ superscript, _ subscript and ~ overbar markers.

    Each marker takes the next character, or a braced group: E~, ~{E}, p^s_t,
    p^{s*}_t. The overbar is what carries the thesis notation for the nominal
    capacities E-bar and P-bar.
    """
    out=[]; i=0
    while i<len(s):
        ch=s[i]
        if ch in '^_~':
            kind={'^':'sup','_':'sub','~':'bar'}[ch]; i+=1
            if i<len(s) and s[i]=='{':
                j=s.index('}',i); out.append((s[i+1:j],kind)); i=j+1
            else: out.append((s[i],kind)); i+=1
        else:
            j=i
            while j<len(s) and s[j] not in '^_~': j+=1
            out.append((s[i:j],'main')); i=j
    return out

def _rich_w(tok,size):
    return sum(stringWidth(t,FONT,size if k in ('main','bar') else size*0.68)
               for t,k in tok)

def tw(text,size,rich=False):
    """Width of `text` at nominal `size`, measured at the active font scale."""
    s=size*FS
    return _rich_w(_rich_tokens(text),s) if rich else stringWidth(text,FONT,s)

def _lines(c,cx,cy,lines,size,color,rich=False):
    if rich:
        lead=size*FS*1.16; s=cy+(len(lines)-1)*lead/2.0
        for i,ln in enumerate(lines): _rich(c,cx,s-i*lead,ln,size,color)
        return
    size=size*FS
    lead=size*1.16; c.setFont(FONT,size); c.setFillColor(color)
    s=cy+(len(lines)-1)*lead/2.0
    for i,ln in enumerate(lines): c.drawCentredString(cx,s-i*lead-size*0.35,ln)

def _rich(c,cx,cy,s,size,color):
    size=size*FS
    tok=_rich_tokens(s); x=cx-_rich_w(tok,size)/2.0; c.setFillColor(color)
    for t,k in tok:
        if k in ('main','bar'): c.setFont(FONT,size); dy=-size*0.35
        elif k=='sub': c.setFont(FONT,size*0.68); dy=-size*0.35-size*0.16
        else: c.setFont(FONT,size*0.68); dy=-size*0.35+size*0.30
        c.drawString(x,cy+dy,t)
        w=stringWidth(t,FONT,size if k in ('main','bar') else size*0.68)
        if k=='bar':
            inset=size*0.06
            c.setStrokeColor(color); c.setLineWidth(size*0.09); c.setLineCap(1)
            c.line(x+inset,cy+dy+size*0.80,x+w-inset,cy+dy+size*0.80)
            c.setLineCap(0); c.setFillColor(color)
        x+=w

def _loop(c,cx,cy,size,left="0, 2",right="1"):
    size=size*FS
    aw,gap=11*FS,3*FS; c.setFont(FONT,size); c.setFillColor(TAG)
    wl=stringWidth(left,FONT,size); wr=stringWidth(right,FONT,size)
    x=cx-(wl+gap+aw+gap+wr)/2.0; c.drawString(x,cy-size*0.35,left)
    ax0=x+wl+gap; ay=cy-size*0.05; c.setStrokeColor(TAG); c.setLineWidth(1.1*FS)
    c.line(ax0,ay,ax0+aw-3*FS,ay)
    p=c.beginPath(); p.moveTo(ax0+aw,ay); p.lineTo(ax0+aw-4*FS,ay+2.4*FS)
    p.lineTo(ax0+aw-4*FS,ay-2.4*FS); p.close(); c.drawPath(p,fill=1,stroke=0)
    c.drawString(ax0+aw+gap,cy-size*0.35,right)

# -- Derived box sizes ------------------------------------------------------
def bw(*items):
    """Box width from (text, nominal_size) or (text, nominal_size, 'r') items."""
    widest=max(tw(it[0],it[1],rich=(len(it)>2)) for it in items)
    return max(widest,MIN_BOX*FS)+PAD_BOX

def h_io():    return 26*FS+12
def h_slice(): return 28*FS+12
def h_comp():  return 34*FS+12
def h_optz():  return 38*FS+12

def hstub(w): return w/2.0+STUB
def vstub(h): return h/2.0+STUB

def pitch_x(widths):
    """Column pitch: widest neighbouring pair on the top row, plus clearance."""
    pairs=[(widths[i]+widths[i+1])/2.0 for i in range(len(widths)-1)]
    return max(pairs)+SKEW+CLEAR_H

def pitch_y(): return h_comp()+CLEAR_V

# -- Node primitives --------------------------------------------------------
def comp(c,cx,cy,w,h,fill,name,tag=None,loop=False,rich=False):
    c.setFillColor(fill)
    if fill is OPT: c.roundRect(cx-w/2,cy-h/2,w,h,h/2.6,fill=1,stroke=0)
    else: c.rect(cx-w/2,cy-h/2,w,h,fill=1,stroke=0)
    if loop: _loop(c,cx,cy+h/2-12*FS,S_LOOP); _lines(c,cx,cy-6*FS,name,S_BODY,white,rich)
    elif tag: _lines(c,cx,cy+h/2-11*FS,[tag],S_TAG,TAG); _lines(c,cx,cy-6*FS,name,S_BODY,white,rich)
    else: _lines(c,cx,cy,name,S_BODY,white,rich)

def _para(c,cx,cy,w,h,skew=SKEW):
    p=c.beginPath()
    p.moveTo(cx-w/2-skew/2,cy-h/2); p.lineTo(cx+w/2-skew/2,cy-h/2)
    p.lineTo(cx+w/2+skew/2,cy+h/2); p.lineTo(cx-w/2+skew/2,cy+h/2); p.close()
    c.setFillColor(DATA); c.drawPath(p,fill=1,stroke=0)

def dnode_io(c,cx,cy,w,l1,l2_rich=None,rich=False):
    """external input/output node: title line, optionally + a variables line."""
    _para(c,cx,cy,w,h_io())
    if l2_rich is not None:
        _lines(c,cx,cy+6*FS,[l1],S_IO_T,white); _rich(c,cx,cy-6*FS,l2_rich,S_IO,white)
    else:
        _lines(c,cx,cy,[l1],S_IO,white,rich)

def dnode_slice(c,cx,cy,w,title,vars_rich):
    """titled dispatch-variable slice: Scheme-A title + rich variables."""
    _para(c,cx,cy,w,h_slice())
    _lines(c,cx,cy+10*FS,[title],S_TITLE,TAG); _rich(c,cx,cy-6*FS,vars_rich,S_VARS,white)

def dline(c,pts):
    c.setStrokeColor(DLINE); c.setLineWidth(LW_DATA*FS); c.setLineJoin(1)
    p=c.beginPath(); p.moveTo(*pts[0])
    for q in pts[1:]: p.lineTo(*q)
    c.drawPath(p,stroke=1,fill=0)

def pline(c,pts):
    c.setStrokeColor(PLINE); c.setLineWidth(LW_PROC*FS); c.setLineJoin(1)
    p=c.beginPath(); p.moveTo(*pts[0])
    for q in pts[1:]: p.lineTo(*q)
    c.drawPath(p,stroke=1,fill=0)

# -- Geometry tables --------------------------------------------------------
def geom_lp():
    g={}
    g['W_IN']   = bw(("~E, ~P (fixed)",S_IO,'r'),("Wind, price",S_IO))
    g['W_OUT']  = bw(("Optimal dispatch",S_IO_T),("p^{s*}_t, p^{c*}_t, e^*_t",S_IO,'r'))
    g['W_DVO']  = bw(("1: Power dispatch",S_TITLE),("p^s_t, p^c_t",S_VARS,'r'))
    g['W_DVC']  = bw(("1: Power + energy state",S_TITLE),("p^s_t, p^c_t, e_t",S_VARS,'r'))
    g['W_OBJ']  = bw(("Objective",S_BODY),("(revenue)",S_BODY))
    g['W_CON']  = bw(("Constraints",S_BODY),)
    g['W_OPTZ'] = bw(("LP optimizer",S_BODY),)
    g['PX']     = pitch_x([g['W_OPTZ'],g['W_DVO'],g['W_DVC']])
    g['PY']     = pitch_y()
    return g

def geom_nlp():
    g={}
    g['W_IN']   = bw(("~E, ~P (fixed)",S_IO,'r'),("Wind, price",S_IO))
    g['W_OUT']  = bw(("Optimal dispatch",S_IO_T),("p^{s*}_t, p^{c*}_t, e^*_t",S_IO,'r'))
    g['W_DVO']  = bw(("1: Power dispatch",S_TITLE),("p^s_t, p^c_t",S_VARS,'r'))
    g['W_DVC']  = bw(("1: Power + energy state",S_TITLE),("p^s_t, p^c_t, e_t",S_VARS,'r'))
    g['W_DVG']  = bw(("1: Energy trajectory",S_TITLE),("e_t",S_VARS,'r'))
    g['W_OBJ']  = bw(("Objective",S_BODY),("(rev. - deg.)",S_BODY))
    g['W_CON']  = bw(("Constraints",S_BODY),)
    g['W_GRD']  = bw(("Gradient",S_BODY),("of objective",S_BODY))
    g['W_RAIN'] = bw(("Rainflow + \u03a6",S_BODY),("(cycles from e_t)",S_BODY,'r'))
    g['W_OPTZ'] = bw(("NLP optimizer",S_BODY),)
    g['PX']     = pitch_x([g['W_OPTZ'],g['W_DVO'],g['W_DVC'],g['W_DVG']])
    g['PY']     = pitch_y()
    return g

def grid(nc,nr,px,py,ml=240,mr=110,mtb=110):
    def cell(col,row): return ml+col*px, mtb+(nr-1-row)*py
    return cell, ml+mr+(nc-1)*px, mtb*2+(nr-1)*py

# -- LP ---------------------------------------------------------------------
def build_lp(path, pagesize=None, translate=(0.0,0.0), fs=None):
    global FS
    FS = FS_LP if fs is None else fs
    g=geom_lp()
    cell,W,H=grid(3,3,g['PX'],g['PY'])
    c=canvas.Canvas(str(path),pagesize=pagesize or (W,H)); c.translate(*translate)
    opt=cell(0,0); obj=cell(1,1); con=cell(2,2)
    dv_obj=cell(1,0); dv_con=cell(2,0)
    out_off=g['W_OPTZ']/2+GAP_OUT+g['W_OUT']/2
    top_off=h_io()/2+h_optz()/2+GAP_TOP
    inp=(opt[0],opt[1]+top_off); xstar=(opt[0]-out_off,opt[1])
    winp=(dv_obj[0],dv_obj[1]+h_io()/2+h_slice()/2+GAP_TOP)
    dline(c,[(inp[0],inp[1]-vstub(h_io())),(opt[0],opt[1]+vstub(h_optz()))])
    dline(c,[(opt[0]-hstub(g['W_OPTZ']),opt[1]),(xstar[0]+hstub(g['W_OUT']),xstar[1])])
    dline(c,[(opt[0]+hstub(g['W_OPTZ']),opt[1]),(dv_obj[0],dv_obj[1])])
    dline(c,[(dv_obj[0]+hstub(g['W_DVO']),dv_obj[1]),(dv_con[0],dv_con[1])])
    dline(c,[(winp[0],winp[1]-vstub(h_io())),(obj[0],obj[1]+vstub(h_comp()))])
    dline(c,[(dv_con[0],dv_con[1]-vstub(h_slice())),(con[0],con[1]+vstub(h_comp()))])
    dline(c,[(obj[0]-hstub(g['W_OBJ']),obj[1]),(opt[0],obj[1]),(opt[0],opt[1]-vstub(h_optz()))])
    dline(c,[(con[0]-hstub(g['W_CON']),con[1]),(opt[0],con[1]),(opt[0],opt[1]-vstub(h_optz()))])
    pline(c,[(opt[0],opt[1]),(obj[0],opt[1]),(obj[0],obj[1]),(con[0],obj[1]),
             (con[0],con[1]),(opt[0],con[1]),(opt[0],opt[1])])
    lx=opt[0]+hstub(g['W_OPTZ'])+18*FS
    _lines(c,lx,obj[1]+10*FS,["2: f"],S_EDGE,GREY)
    _lines(c,lx,con[1]+10*FS,["2: c"],S_EDGE,GREY)
    dnode_io(c,inp[0],inp[1],g['W_IN'],"~E, ~P (fixed)",rich=True)
    dnode_io(c,winp[0],winp[1],g['W_IN'],"Wind, price")
    dnode_io(c,xstar[0],xstar[1],g['W_OUT'],"Optimal dispatch",l2_rich="p^{s*}_t, p^{c*}_t, e^*_t")
    dnode_slice(c,dv_obj[0],dv_obj[1],g['W_DVO'],"1: Power dispatch","p^s_t, p^c_t")
    dnode_slice(c,dv_con[0],dv_con[1],g['W_DVC'],"1: Power + energy state","p^s_t, p^c_t, e_t")
    comp(c,obj[0],obj[1],g['W_OBJ'],h_comp(),ANLY,["Objective","(revenue)"],tag="1:")
    comp(c,con[0],con[1],g['W_CON'],h_comp(),ANLY,["Constraints"],tag="1:")
    comp(c,opt[0],opt[1],g['W_OPTZ'],h_optz(),OPT,["LP optimizer"],loop=True)
    c.showPage(); c.save(); return W,H

# -- NLP --------------------------------------------------------------------
def build_nlp(path, pagesize=None, translate=(0.0,0.0), fs=None):
    global FS
    FS = FS_NLP if fs is None else fs
    g=geom_nlp()
    cell,W,H=grid(4,4,g['PX'],g['PY'])
    c=canvas.Canvas(str(path),pagesize=pagesize or (W,H)); c.translate(*translate)
    opt=cell(0,0); obj=cell(1,1); con=cell(2,2); grd=cell(3,3); rain=cell(3,1)
    dv_obj=cell(1,0); dv_con=cell(2,0); dv_grd=cell(3,0)
    out_off=g['W_OPTZ']/2+GAP_OUT+g['W_OUT']/2
    top_off=h_io()/2+h_optz()/2+GAP_TOP
    inp=(opt[0],opt[1]+top_off); xstar=(opt[0]-out_off,opt[1])
    winp=(dv_obj[0],dv_obj[1]+h_io()/2+h_slice()/2+GAP_TOP)
    dline(c,[(inp[0],inp[1]-vstub(h_io())),(opt[0],opt[1]+vstub(h_optz()))])
    dline(c,[(opt[0]-hstub(g['W_OPTZ']),opt[1]),(xstar[0]+hstub(g['W_OUT']),xstar[1])])
    dline(c,[(opt[0]+hstub(g['W_OPTZ']),opt[1]),(dv_obj[0],dv_obj[1])])
    dline(c,[(dv_obj[0]+hstub(g['W_DVO']),dv_obj[1]),(dv_con[0],dv_con[1])])
    dline(c,[(dv_con[0]+hstub(g['W_DVC']),dv_con[1]),(dv_grd[0],dv_grd[1])])
    dline(c,[(winp[0],winp[1]-vstub(h_io())),(obj[0],obj[1]+vstub(h_comp()))])
    dline(c,[(dv_con[0],dv_con[1]-vstub(h_slice())),(con[0],con[1]+vstub(h_comp()))])
    dline(c,[(dv_grd[0],dv_grd[1]-vstub(h_slice())),(grd[0],grd[1]+vstub(h_comp()))])
    dline(c,[(obj[0]-hstub(g['W_OBJ']),obj[1]),(opt[0],obj[1]),(opt[0],opt[1]-vstub(h_optz()))])
    dline(c,[(con[0]-hstub(g['W_CON']),con[1]),(opt[0],con[1]),(opt[0],opt[1]-vstub(h_optz()))])
    dline(c,[(grd[0]-hstub(g['W_GRD']),grd[1]),(opt[0],grd[1]),(opt[0],opt[1]-vstub(h_optz()))])
    dline(c,[(rain[0],rain[1]+vstub(h_comp())),(rain[0],obj[1]),(obj[0]+hstub(g['W_OBJ']),obj[1])])
    dline(c,[(rain[0],rain[1]-vstub(h_comp())),(grd[0],grd[1]+vstub(h_comp()))])
    pline(c,[(opt[0],opt[1]),(obj[0],opt[1]),(obj[0],obj[1]),(con[0],obj[1]),
             (con[0],con[1]),(grd[0],con[1]),(grd[0],grd[1]),(opt[0],grd[1]),(opt[0],opt[1])])
    lx=opt[0]+hstub(g['W_OPTZ'])+18*FS
    _lines(c,lx,obj[1]+10*FS,["2: f"],S_EDGE,GREY)
    _lines(c,lx,con[1]+10*FS,["2: c"],S_EDGE,GREY)
    _lines(c,lx+4*FS,grd[1]+10*FS,["2: df/de"],S_EDGE,GREY)
    fw=max(g['W_GRD']/2+16, tw("gradient is discontinuous here",S_FLAG)/2+4)
    c.setStrokeColor(FLAG); c.setLineWidth(1.3*FS); c.setDash(3*FS,3*FS)
    c.line(grd[0]-fw,grd[1]-h_comp()/2-9*FS,grd[0]+fw,grd[1]-h_comp()/2-9*FS); c.setDash()
    _lines(c,grd[0],grd[1]-h_comp()/2-19*FS,["gradient is discontinuous here"],S_FLAG,FLAG)
    dnode_io(c,inp[0],inp[1],g['W_IN'],"~E, ~P (fixed)",rich=True)
    dnode_io(c,winp[0],winp[1],g['W_IN'],"Wind, price")
    dnode_io(c,xstar[0],xstar[1],g['W_OUT'],"Optimal dispatch",l2_rich="p^{s*}_t, p^{c*}_t, e^*_t")
    dnode_slice(c,dv_obj[0],dv_obj[1],g['W_DVO'],"1: Power dispatch","p^s_t, p^c_t")
    dnode_slice(c,dv_con[0],dv_con[1],g['W_DVC'],"1: Power + energy state","p^s_t, p^c_t, e_t")
    dnode_slice(c,dv_grd[0],dv_grd[1],g['W_DVG'],"1: Energy trajectory","e_t")
    comp(c,obj[0],obj[1],g['W_OBJ'],h_comp(),ANLY,["Objective","(rev. - deg.)"],tag="1:")
    comp(c,con[0],con[1],g['W_CON'],h_comp(),ANLY,["Constraints"],tag="1:")
    comp(c,grd[0],grd[1],g['W_GRD'],h_comp(),GRAD,["Gradient","of objective"],tag="1:")
    comp(c,rain[0],rain[1],g['W_RAIN'],h_comp(),DEG,
         ["Rainflow + \u03a6","(cycles from e_t)"],tag="1:",rich=True)
    comp(c,opt[0],opt[1],g['W_OPTZ'],h_optz(),OPT,["NLP optimizer"],loop=True)
    c.showPage(); c.save(); return W,H

# -- Export: tight-cropped PDF (vector) and PNG -----------------------------
PAD_PT     = 6.0     # left / right / bottom margin
TOP_PAD_PT = 14.0    # margin above the top box
MEAS_M     = 90.0    # scratch margin so nothing touches the page edge
DPI        = 300     # raster resolution for the PNG


def _render_pdf_to_png(pdf_path, png_path, dpi):
    """Render page 1 of a PDF to PNG.

    Prefers PyMuPDF (``pip install pymupdf``) because it is pure-pip and needs
    no external binary, so it runs unchanged on Windows. Falls back to poppler's
    ``pdftoppm`` if PyMuPDF is not installed.
    """
    try:
        try:
            import pymupdf              # PyMuPDF 1.24.3 and later
        except ImportError:
            import fitz as pymupdf      # older releases expose it as fitz
        doc = pymupdf.open(str(pdf_path))
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(dpi/72.0, dpi/72.0), alpha=False)
        pix.save(str(png_path)); doc.close(); return
    except ImportError:
        pass
    import shutil, subprocess
    if shutil.which("pdftoppm") is None:
        raise RuntimeError(
            "No PDF renderer found for PNG export. Install PyMuPDF:\n"
            "    pip install pymupdf\n"
            "(or install poppler and put pdftoppm on PATH).")
    stem = str(png_path)[:-4] if str(png_path).endswith(".png") else str(png_path)
    subprocess.run(["pdftoppm","-png","-r",str(dpi),"-singlefile",
                    str(pdf_path),stem], check=True)


def emit(build_fn, pdf_out, png_out, fs=None,
         pad=PAD_PT, top_pad=TOP_PAD_PT, dpi=DPI):
    """Write a tight-cropped PDF (regenerated at the cropped size, still vector)
    and a matching PNG. Returns the cropped page size in points."""
    scratch_pdf = str(pdf_out)+".scratch.pdf"
    scratch_png = str(pdf_out)+".scratch.png"

    # 1. Draw once on an oversized page so nothing clips, then render it.
    W,H = build_fn(scratch_pdf, fs=fs)
    build_fn(scratch_pdf, pagesize=(W+2*MEAS_M, H+2*MEAS_M),
             translate=(MEAS_M,MEAS_M), fs=fs)
    _render_pdf_to_png(scratch_pdf, scratch_png, dpi)

    # 2. Content bounding box from the rendered pixels (non-white).
    gray = np.asarray(Image.open(scratch_png).convert("L"))
    ys,xs = np.where(gray < 250)
    px0,px1 = int(xs.min()), int(xs.max())
    py0,py1 = int(ys.min()), int(ys.max())
    s = dpi/72.0
    page_h = H+2*MEAS_M
    x0,x1 = px0/s, (px1+1)/s
    y0,y1 = page_h-(py1+1)/s, page_h-py0/s

    # 3. Vector PDF: regenerate at the cropped size with matching margins.
    new_W = (x1-x0)+2*pad
    new_H = (y1-y0)+pad+top_pad
    if OUTPUT in ("pdf","both"):
        build_fn(pdf_out, pagesize=(new_W,new_H),
                 translate=(pad+MEAS_M-x0, pad+MEAS_M-y0), fs=fs)

    # 4. PNG: crop the scratch raster to the same box (lossless).
    if OUTPUT in ("png","both"):
        pad_px, top_px = int(round(pad*s)), int(round(top_pad*s))
        img = Image.open(scratch_png)
        img.crop((max(px0-pad_px,0), max(py0-top_px,0),
                  min(px1+1+pad_px,img.width),
                  min(py1+1+pad_px,img.height))).save(str(png_out))

    os.remove(scratch_pdf); os.remove(scratch_png)
    return new_W, new_H


# -- Diagnostics ------------------------------------------------------------
def report_clearances(g):
    """Drawn length of the two short connectors around the optimizer.

    These are the only connectors whose length is set by a single offset rather
    than a full row or column pitch, so they vanish first if the layout is
    retuned. Below about 12 pt they do not read as a line at print size.
    """
    out_off = g['W_OPTZ']/2+GAP_OUT+g['W_OUT']/2
    for name,length in [("input -> optimizer", GAP_TOP),
                        ("optimizer -> output", out_off-g['W_OPTZ']/2-g['W_OUT']/2)]:
        print(f"  {'OK' if length>=12 else 'TOO SHORT':9s} {name:22s} {length:6.1f} pt")


def sweep(textwidth=418.0, values=(1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,4.0)):
    """Print the on-page text size each font scale produces for the NLP figure.

    `textwidth` is the LaTeX \\textwidth in points the figure will be scaled to
    fit. The figure is built and measured at each scale, then the shrink factor
    textwidth / figure_width is applied to the nominal font sizes.
    """
    import tempfile
    print(f"NLP font-scale sweep, target width {textwidth:.0f} pt")
    print(f"{'FS':>5} {'fig W':>8} {'fig H':>8} {'shrink':>7} "
          f"{'body pt':>8} {'title pt':>9} {'aspect':>7}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for v in values:
            W,H = emit(build_nlp, td/"a.pdf", td/"a.png", fs=v, dpi=100)
            k = textwidth/W
            print(f"{v:5.2f} {W:8.1f} {H:8.1f} {k:7.3f} "
                  f"{S_BODY*v*k:8.2f} {S_TITLE*v*k:9.2f} {W/H:7.2f}")


if __name__ == "__main__":
    if OUTPUT not in ("png","pdf","both"):
        raise ValueError(f'OUTPUT must be "png", "pdf" or "both", not {OUTPUT!r}')
    out = Path(__file__).parent
    if "--sweep" in sys.argv:
        sweep(); sys.exit(0)
    FS = FS_NLP
    print("NLP connector clearances:"); report_clearances(geom_nlp())
    w_lp,h_lp   = emit(build_lp,  out/"dispatch_lp_xdsm.pdf",  out/"dispatch_lp_xdsm.png")
    w_nlp,h_nlp = emit(build_nlp, out/"monolithic_nlp_xdsm.pdf", out/"monolithic_nlp_xdsm.png")
    print(f"dispatch_lp_xdsm     FS={FS_LP:.2f}   {w_lp:.1f} x {h_lp:.1f} pt")
    print(f"monolithic_nlp_xdsm  FS={FS_NLP:.2f}   {w_nlp:.1f} x {h_nlp:.1f} pt")
    print(f"Wrote {OUTPUT} for both figures in", out)