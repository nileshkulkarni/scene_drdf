
//math fun
function linspace(zmin, zmax, zsamp){
    let zs = new Array();
    let zdelta = (zmax-zmin)/zsamp;
    for(let i=0; i < zsamp; i++){
        zs.push(zmin+i*(zmax-zmin)/(zsamp-1));
    }
    return zs;
}


SIGTRUNC = 4;
SQRT2PI = Math.sqrt(2*Math.PI);
//standard normal pdf
function snPDF(x){ 
    if(Math.abs(x) > SIGTRUNC){ return 0; }
    return Math.exp(-0.5*(x*x))/SQRT2PI;
}

//standard normal cdf
function snCDFCompute(x){
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
    if(x/sigma > SIGTRUNC){ return 0; } // past the truncation = 0
    let evalAt = linspace(x,4,50);
    let f = function(z){ return (z/sigma)*snPDF(z/sigma); }
    let s = 0; //sum term
    let last = 0; // previous iteration's value
    for(let i=0; i < evalAt.length-1; i++){
        let a = evalAt[i]; let b = evalAt[i+1];
        let fa = last; let fb = f(b);
        //doing fancier quadrature is better than more function calls
        s += (b-a)*(fa + 4*f((a+b)/2) + fb)/6;
        last = fb;
    }
    return s;
}


let SNCDFTAB = new Array();
let SNPEXPECTTAB = new Array();
const SNSAMP = 100;
function snCDF(x){
    if(SNCDFTAB.length == 0){
        //fill in the interpolation table
        for(let i=0; i < SNSAMP; i++){
            SNCDFTAB.push(snCDFCompute(i/(SNSAMP-1)*SIGTRUNC*2-SIGTRUNC));
        }
    }
    if(x < -SIGTRUNC){ return 0; }
    else if(x > SIGTRUNC){ return 1; } 
    let ind = (x+SIGTRUNC)/(2*SIGTRUNC)*(SNSAMP-1);
    let below = Math.floor(ind); let above = Math.ceil(ind);
    let belowVal = SNCDFTAB[below]; let aboveVal = SNCDFTAB[above];
    let alpha = above-ind;
    return alpha*belowVal+(1-alpha)*aboveVal;
}

function drdf(z){ return (z < 0.5) ? -z : 1-z; }
function urdf(z){ return (z < 0.5) ? Math.abs(z) : 1-z; }
function srdf(z){ return (z < 0.5) ? -z : z-1; }

function getNPDF(mu, sigma){
    return function(z){ return snPDF((z-mu)/sigma); }
}

function getNCDF(mu,sigma){
    return function(z){ return snCDF((z-mu)/sigma); }
}

function getExpectedDRDF(sigma){
    return function(z){ return snCDF((z-0.5)/sigma)-z; }
}

function getExpectedORF(sigma, rad){
    return function(z){ return snCDF((z+rad)/sigma) - snCDF((z-rad)/sigma); } 
}

function getExpectedURDF(sigma){
    return function(z){ return z*snCDF(z/sigma) - z*(1-snCDF(z/sigma)) + 2 * nPartialExpect(z, sigma) + (1-2*z)*snCDF((z-0.5)/sigma) - nPartialExpect(z-0.5,sigma); }
}

function getExpectedSRDF(sigma){
    return function(z){ return -z+(2*z-1)*snCDF((z-0.5)/sigma) + 2*nPartialExpect(z-0.5,sigma); }
}


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
    ctx.clearRect(0,0,can.width,can.height);

    ctx.lineWidth=1;
    let minorYTicks = [0.5,0.25,0,-0.25,-0.5];
    let minorXTicks = [-0.5,0,0.5,1,1.5];
    ctx.strokeStyle='#aaa'
    for(let i=0; i < minorYTicks.length; i++){
        drawPath(ctx, cvf, [[-5,minorYTicks[i]], [5,minorYTicks[i]]]);
    }
    for(let i=0; i < minorXTicks.length; i++){
        drawPath(ctx, cvf, [[minorXTicks[i],-5], [minorXTicks[i],5]]);
    }


    ctx.strokeStyle='black';
    ctx.lineWidth=4;

    drawPath(ctx, cvf, [[0,-10],[0,10]]);
    drawPath(ctx, cvf, [[-10,0],[10,0]]);

    ctx.setLineDash([10,10]);
    ctx.strokeStyle='#666';
    drawPath(ctx, cvf, [[1.0,-10],[1.0,10]]);
    ctx.setLineDash([])


    let labelW = 50; let labelH = 25;
    function extraBox(text,loc){
        let locCan = cvf(loc);
        ctx.strokeStyle='black';
        ctx.fillStyle='white';
        ctx.fillRect(locCan[0]-labelW/2,locCan[1]-labelH/2,labelW,labelH);
        ctx.strokeRect(locCan[0]-labelW/2,locCan[1]-labelH/2,labelW,labelH);

        ctx.font = '20px Serif';
        ctx.textAlign='center';
        ctx.fillStyle='black';
        ctx.fillText(text,locCan[0],locCan[1]+labelH*0.20,labelW);
    }
    extraBox('μ',[0,-0.6]);
    extraBox('μ+1',[1,-0.6]);
}

function drawYTicks(canvasId){
    let can = document.getElementById(canvasId);
    
    if(can.getAttribute('data-drawn') == "1"){ return; } 

    let ctx = can.getContext('2d');
    let cvf = canvasToCVF(canvasId);
 
    let labelW = 50; let labelH = 25;
    function extraBox(text,loc){
        let locCan = cvf(loc);
        ctx.font = '24px Serif';
        ctx.textAlign='center';
        ctx.fillStyle='black';
        ctx.fillText(text,locCan[0],locCan[1]+labelH*0.15,labelW);
    }
    const labelsAndLocs = [["0.5",0.5],["0.25",0.25],["0",0],["-0.25",-0.25],["-0.5",-0.5]];
    for(let i=0;i < labelsAndLocs.length; i++){
        extraBox(labelsAndLocs[i][0],[0,labelsAndLocs[i][1]]);
    }
    can.setAttribute("data-drawn","1");
}

function drawLegend(canvasId){
    let can = document.getElementById(canvasId);
    
    if(can.getAttribute('data-drawn') == "1"){ return; } 
    console.log('coming to draw legend')
    let ctx = can.getContext('2d');
    let ch = can.height; let cw = can.width;

    if(0){
        let N = toLegend.length;
        for(let i=0; i < toLegend.length; i++){
            //unpack
            let textLabel = toLegend[i][0];
            let lineColor = toLegend[i][1];
            let lineDash = toLegend[i][2];

            let cx = (i+1)/(N+1)*cw;

            ctx.strokeStyle=lineColor;
            console.log(ctx.strokeStyle);
            ctx.setLineDash(lineDash);
            ctx.lineWidth=4;
            ctx.beginPath();
            ctx.moveTo(cx-30,ch/4);
            ctx.lineTo(cx+30,ch/4);
            ctx.stroke();

        }
    }
    
    let groups = ["Uncertainty","SRDF","URDF","DRDF"];

    let toLegend = [[["PDF","#785EF0",[],4]],
        [["Actual","#648FFF",[10,10],2],["Expected","#648FFF",[],4]],
        [["Actual","#DC267F",[10,10],2],["Expected","#DC267F",[],4]],
        [["Actual","#FE6100",[10,10],2],["Expected","#FE6100",[],4]],
    ];

    for(let i = 0; i < groups.length; i++){
        //group label

        let cx = cw*(i+0.5)/(groups.length);
        let blockSize = cw/(groups.length);

        ctx.font = "20px Serif";
        ctx.fillStyle = "black";
        ctx.textAlign= "center";
        ctx.textBaseline = "middle";
        ctx.fillText(groups[i], cx, ch/6);


        let legendBlock = toLegend[i];
        for(let j = 0; j < legendBlock.length; j++){

            let cy = (ch/6)+5*ch/6*(j+1)/(legendBlock.length+1);

            let textLabel = legendBlock[j][0];
            let lineColor = legendBlock[j][1];
            let lineDash = legendBlock[j][2];
            let lineWidth = legendBlock[j][3];

            ctx.lineWidth = lineWidth;
            ctx.setLineDash(lineDash);
            ctx.strokeStyle = lineColor;

            ctx.beginPath();
            ctx.moveTo(cx-0.35*blockSize, cy);
            ctx.lineTo(cx-0.10*blockSize, cy);
            console.log(blockSize);
            ctx.fillColor = "black";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(textLabel, cx+0.15*blockSize, cy);
            ctx.stroke();

        }
    }
    





    can.setAttribute("data-drawn","1");
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
    let sigma = Number(document.getElementById('val').value);

    drawYTicks("annot")
    drawLegend("legend");

    drawAxes("pdf");
    drawFunction("pdf",getNPDF(0,sigma),"#785EF0");

    drawAxes('graphdrdf'); 
    drawFunction("graphdrdf",drdf,'#FE6100',lineWidth=2,lineDash=[10,10]);
    drawFunction("graphdrdf",getExpectedDRDF(sigma),'#FE6100');

    drawAxes('graphurdf'); 
    drawFunction("graphurdf",urdf,'#DC267F',lineWidth=2,lineDash=[10,10]);
    drawFunction("graphurdf",getExpectedURDF(sigma),'#DC267F');

    drawAxes('graphsrdf');
    drawFunction("graphsrdf",srdf,'#648FFF',lineWidth=2,lineDash=[10,10]);
    drawFunction("graphsrdf",getExpectedSRDF(sigma),'#648FFF');

}

window.onload=draw;