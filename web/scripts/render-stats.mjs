// Reduce rendered pixels locally to numbers; never return or save clinical pixels.
export async function readMPRStats(page) {
  return page.evaluate(() => [0,1,2].map(i => {
    const canvas=document.querySelector(`[data-testid="viewport-${i}"] canvas`);
    if(!canvas)return {plane:i,missing:true};
    const ctx=canvas.getContext('2d');
    if(!ctx)return {plane:i,no2DContext:true,width:canvas.width,height:canvas.height};
    const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
    let gray=0,colored=0;const levels=new Set();
    for(let j=0;j<pixels.length;j+=16){
      const r=pixels[j],g=pixels[j+1],b=pixels[j+2];
      if(r>25&&Math.max(r,g,b)-Math.min(r,g,b)<=3){gray++;levels.add(r);}
      if(Math.max(r,g,b)>65&&Math.max(r,g,b)-Math.min(r,g,b)>25)colored++;
    }
    return {plane:i,width:canvas.width,height:canvas.height,gray,grayLevels:levels.size,colored};
  }));
}
