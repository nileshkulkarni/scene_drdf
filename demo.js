function linspace(zmin, zmax, zsamp){
    let zs = new Array();
    let zdelta = (zmax-zmin)/zsamp;
    for(let i=0; i < zsamp; i++){
        zs.push(zmin+i*(zmax-zmin)/(zsamp-1));
    }
    return zs;
}


SIGTRUNC = 4;
//standard normal pdf
function snPDF(x){ 
    if(Math.abs(x) > SIGTRUNC){ return 0; }
    return Math.exp(-0.5*(x*x))/(Math.sqrt(2*Math.PI)); 
}

//standard normal cdf
function snCDF(x){
    //Zelen & Severo (1964) + 
    if(x < -SIGTRUNC){ return 0; } 
    else if(x > SIGTRUNC){ return 1;} 
    const b0 = 0.2316419; const b1 = 0.319381530; const b2 = -0.356563782;
    const b3 = 1.781477937; const b4 = -1.821255978; const b5 = 1.330274429;
    let flip = 0;
    if(x < 0){ flip = 1; x = -x; }
    let t = 1 / (1+b0*x);
    let poly = b1*t + b2*t*t + b3*Math.pow(t,3) + b4*Math.pow(t,4) + b5*Math.pow(t,5);
    let f = 1 - snPDF(x)*poly;
    if(flip){ return 1 - f;
    } else { return f; }
}

//\int_{x}^{\infty} x p(x) dx by quadrature
function nPartialExpect(x,sigma){
    if(x < 0){ x = -x; } // the function is symmetric
    if(x/sigma > SIGTRUNC){ return 0; } // past 8 sigma = 0
    //if(x < 0){ x = -x; sign= -1}
    let evalAt = linspace(x,4,50);
    let fevalAt = new Array();
    let f = function(z){ return (z/sigma)*snPDF(z/sigma); }
    let s = 0; //sum term
    let last = 0; // previous iteration's value
    for(let i=0; i < evalAt.length-1; i++){
        let a = evalAt[i]; let b = evalAt[i+1];
        let fa = last;
        let fb = f(b);
        let xp = (b-a) / 6;
        s += xp*(fa + 4*f((a+b)/2) + fb);
        last = fb;
    }
    return s;
}

function getNPDF(mu, sigma){
    return function(z){ return snPDF((z-mu)/sigma); }
}

function getNCDF(mu,sigma){
    return function(z){ return snCDF((z-mu)/sigma); }
}

function getExpectedDRDF(sigma){
    return function(z){ return snCDF((z-0.5)/sigma)-z; }
}

function getExpectedORF(sigma){
    return function(z){ return snCDF((z+0.2)/sigma) - snCDF((z-0.2)/sigma); } 
}

function getExpectedURDF(sigma){
    return function(z){ return z*snCDF(z/sigma) - z*(1-snCDF(z/sigma)) + 2 * nPartialExpect(z, sigma) + (1-2*z)*snCDF((z-0.5)/sigma) - nPartialExpect(z-0.5,sigma); }
}

function getExpectedSRDF(sigma){
    return function(z){ return -z+(2*z-1)*snCDF((z-0.5)/sigma) + 2*nPartialExpect(z-0.5,sigma); }
}

function drdf(z){ return (z < 0.5) ? -z : 1-z; }
function urdf(z){ return (z < 0.5) ? Math.abs(z) : 1-z; }
function srdf(z){ return (z < 0.5) ? -z : z-1; }

function getXYToCanvas(canSize, canGeom){
    //given:
    //  canSize (width,height) and
    //  canGeom (xmin,xmax,ymin,ymax)
    //return function that converts x/y to canvas coords
    return function convert(p){
        let locx = (p[0]-canGeom[0])/(canGeom[1]-canGeom[0])*canSize[0];
        let locy = canSize[1]-(p[1]-canGeom[2])/(canGeom[3]-canGeom[2])*canSize[1];
        return [locx, locy];
    }
}

function canvasToCVF(canvasId){
    let can = document.getElementById(canvasId);
    let lims = can.getAttribute("data-lims").split(",").map(Number);
    return getXYToCanvas([can.width, can.height],lims);
}

function canvasToLims(canvasId){
    return document.getElementById(canvasId).getAttribute("data-lims").split(",").map(Number);
}

function drawPath(ctx, cvf, path){
    ctx.beginPath();
    let canPath = path.map(cvf); 
    ctx.moveTo(canPath[0][0], canPath[0][1]);
    for(let i=1; i < canPath.length; i++){
        let p = canPath[i];
        ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
}


function drawAxes(canvasId){
    let can = document.getElementById(canvasId);
    let ctx = can.getContext('2d');
    let cvf = canvasToCVF(canvasId);
    
    ctx.beginPath();
    ctx.strokeStyle='black';
    ctx.lineWidth=2;
    ctx.clearRect(0,0,can.width,can.height);

    drawPath(ctx, cvf, [[0,-10],[0,10]]);
    drawPath(ctx, cvf, [[-10,0],[10,0]]);

    ctx.setLineDash([10,10]);
    ctx.strokeStyle='gray';
    drawPath(ctx, cvf, [[1.0,-10],[1.0,10]]);
    ctx.setLineDash([])

}


function drawFunction(canvasId, fun, strokeStyle, lineWidth=4,lineDash=[]){
    let can = document.getElementById(canvasId);
    let cvf = canvasToCVF(canvasId);
    let lims = canvasToLims(canvasId);
    let fudge = (lims[1]-lims[0])*0.05; //evaluate not just from xmin->xmax but also a little extra
    let zs = linspace(lims[0]-fudge,lims[1]+fudge,can.width*2);
    let path = new Array();
    for(let i = 0; i < zs.length; i++){
        path.push([zs[i], fun(zs[i])])
    }
    let ctx = can.getContext('2d');
    ctx.strokeStyle=strokeStyle;
    ctx.setLineDash(lineDash);
    ctx.lineWidth=lineWidth;
    drawPath(ctx, cvf, path);
}

function draw(){
    let val = document.getElementById('val').value;
    drawAxes('graphdrdf'); drawAxes('graphurdf'); drawAxes('graphsrdf');
    drawAxes("pdf");
    drawFunction("pdf",getNPDF(0,val),"black");
    drawFunction("graphdrdf",drdf,'#FE6100',lineWidth=2,lineDash=[10,10]);
    drawFunction("graphdrdf",getExpectedDRDF(val),'#FE6100');
    drawFunction("graphurdf",urdf,'#DC267F',lineWidth=2,lineDash=[10,10]);
    drawFunction("graphurdf",getExpectedURDF(val),'#DC267F');
    drawFunction("graphsrdf",srdf,'#648FFF',lineWidth=2,lineDash=[10,10]);
    drawFunction("graphsrdf",getExpectedSRDF(val),'#648FFF');
}
window.onload=draw;