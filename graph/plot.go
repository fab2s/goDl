// Training visualization: HTML charts, CSV export, and text logs for epoch trends.
//
// PlotHTML generates a self-contained HTML file with interactive training
// curves — no external dependencies, no npm, no CDN. Open in any browser.
//
//	g.PlotHTML("training.html", "loss", "head_0", "head_1")
//
// ExportTrends writes epoch history to CSV for external analysis tools.
//
//	g.ExportTrends("metrics.csv", "loss", "accuracy")
//
// WriteLog writes a human-readable training log with per-epoch metrics,
// timing, and ETA:
//
//	g.WriteLog("training.log", 100, "loss", "hit_rate")
package graph

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// PlotHTML generates a self-contained HTML file with training curves
// for the specified tags. Tag group names are expanded automatically.
// If no tags are specified, all tags with epoch history are plotted.
//
// The generated file uses inline JavaScript with HTML5 Canvas — no
// external dependencies. Open it in any browser.
//
//	// Plot specific tags.
//	g.PlotHTML("training.html", "loss", "accuracy")
//
//	// Plot a tag group (expands to head_0, head_1, head_2).
//	g.PlotHTML("heads.html", "head")
//
//	// Plot everything.
//	g.PlotHTML("all.html")
func (g *Graph) PlotHTML(path string, tags ...string) error {
	series := g.gatherSeries(tags)
	if len(series) == 0 {
		return fmt.Errorf("graph: no epoch data to plot (call Flush first)")
	}

	data, err := json.Marshal(series)
	if err != nil {
		return fmt.Errorf("graph: marshal plot data: %w", err)
	}

	html := strings.Replace(plotTemplate, "/*DATA*/", string(data), 1)

	p := filepath.Clean(path)
	if err := os.WriteFile(p, []byte(html), 0600); err != nil {
		return fmt.Errorf("graph: write HTML to %s: %w", p, err)
	}
	return nil
}

// PlotTimingsHTML generates a self-contained HTML file with timing
// trend curves. Same as [PlotHTML] but uses timing epoch history
// (from [Graph.CollectTimings] / [Graph.FlushTimings]).
//
//	g.PlotTimingsHTML("timings.html", "encoder", "decoder")
func (g *Graph) PlotTimingsHTML(path string, tags ...string) error {
	series := g.gatherTimingSeries(tags)
	if len(series) == 0 {
		return fmt.Errorf("graph: no timing data to plot (call FlushTimings first)")
	}

	data, err := json.Marshal(series)
	if err != nil {
		return fmt.Errorf("graph: marshal timing data: %w", err)
	}

	html := strings.Replace(plotTemplate, "/*DATA*/", string(data), 1)
	html = strings.Replace(html, "Training Curves", "Timing Trends", 1)

	p := filepath.Clean(path)
	if err := os.WriteFile(p, []byte(html), 0600); err != nil {
		return fmt.Errorf("graph: write HTML to %s: %w", p, err)
	}
	return nil
}

// ExportTrends writes epoch history to a CSV file. Columns are epoch
// number followed by one column per tag. Tag groups are expanded.
// If no tags are specified, all tags with history are exported.
//
//	g.ExportTrends("metrics.csv", "loss", "accuracy")
//
// Output:
//
//	epoch,loss,accuracy
//	1,0.5432,0.7123
//	2,0.4321,0.7856
func (g *Graph) ExportTrends(path string, tags ...string) error {
	series := g.gatherSeries(tags)
	if len(series) == 0 {
		return fmt.Errorf("graph: no epoch data to export (call Flush first)")
	}

	return writeCSV(path, series)
}

// ExportTimingTrends writes timing epoch history to a CSV file.
// Same format as [ExportTrends] but uses timing data.
func (g *Graph) ExportTimingTrends(path string, tags ...string) error {
	series := g.gatherTimingSeries(tags)
	if len(series) == 0 {
		return fmt.Errorf("graph: no timing data to export (call FlushTimings first)")
	}

	return writeCSV(path, series)
}

// WriteLog writes a human-readable training log to a text file.
// Each line shows one epoch with metric values, duration, and ETA.
// Tag group names are expanded. If no tags are specified, all tags
// with epoch history are included.
//
// totalEpochs is the expected total number of epochs (for ETA
// calculation). If 0, ETA is omitted.
//
//	g.WriteLog("training.log", 100, "loss", "hit_rate", "lr")
//
// Output:
//
//	# goDl training log — 2026-03-08T10:30:00Z
//	epoch   1  loss=0.5432  hit_rate=71.23%  lr=0.001000  [1.2s  ETA 1m58s]
//	epoch   2  loss=0.4321  hit_rate=78.56%  lr=0.001000  [1.1s  ETA 1m47s]
func (g *Graph) WriteLog(path string, totalEpochs int, tags ...string) error {
	series := g.gatherSeries(tags)
	if len(series) == 0 {
		return fmt.Errorf("graph: no epoch data to log (call Flush first)")
	}

	var b strings.Builder

	// Header.
	startTime := time.Now()
	if len(g.flushTimes) > 0 {
		startTime = g.flushTimes[0]
	}
	fmt.Fprintf(&b, "# goDl training log — %s\n", startTime.Format(time.RFC3339))

	// Find max epoch count.
	maxLen := 0
	for _, s := range series {
		if len(s.Values) > maxLen {
			maxLen = len(s.Values)
		}
	}

	// Epoch lines.
	for i := range maxLen {
		fmt.Fprintf(&b, "epoch %3d", i+1)
		for _, s := range series {
			if i < len(s.Values) {
				fmt.Fprintf(&b, "  %s=%s", s.Name, formatMetric(s.Values[i]))
			}
		}
		// Per-epoch duration.
		if i < len(g.flushTimes) {
			var dur time.Duration
			if i == 0 {
				dur = g.flushTimes[0].Sub(g.trainingStart)
			} else {
				dur = g.flushTimes[i].Sub(g.flushTimes[i-1])
			}
			if dur > 0 {
				b.WriteString("  [")
				b.WriteString(FormatDuration(dur))
				// ETA.
				if totalEpochs > 0 {
					elapsed := g.flushTimes[i].Sub(g.trainingStart)
					perEpoch := elapsed / time.Duration(i+1)
					remaining := perEpoch * time.Duration(totalEpochs-i-1)
					if remaining > 0 {
						fmt.Fprintf(&b, "  ETA %s", FormatDuration(remaining))
					}
				}
				b.WriteByte(']')
			}
		}
		b.WriteByte('\n')
	}

	p := filepath.Clean(path)
	return os.WriteFile(p, []byte(b.String()), 0600)
}

// FormatDuration formats a duration for training logs:
// <1s → "42ms", <1m → "1.2s", ≥1m → "2m05s".
func FormatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
}

// formatMetric formats a scalar value for log display.
func formatMetric(v float64) string {
	abs := v
	if abs < 0 {
		abs = -abs
	}
	switch {
	case abs == 0:
		return "0"
	case abs < 0.001:
		return fmt.Sprintf("%.2e", v)
	case abs < 1:
		return fmt.Sprintf("%.4f", v)
	case abs < 100:
		return fmt.Sprintf("%.4f", v)
	default:
		return fmt.Sprintf("%.2f", v)
	}
}

// plotSeries holds the data for one line in a chart.
type plotSeries struct {
	Name   string    `json:"name"`
	Values []float64 `json:"values"`
}

// gatherSeries collects epoch history for the given tags (expanding groups).
func (g *Graph) gatherSeries(tags []string) []plotSeries {
	if g.epochHistory == nil {
		return nil
	}

	if len(tags) == 0 {
		// All tags with data.
		for tag := range g.epochHistory {
			tags = append(tags, tag)
		}
		sort.Strings(tags)
	} else {
		tags = g.expandGroups(tags)
	}

	series := make([]plotSeries, 0, len(tags))
	for _, tag := range tags {
		vals, ok := g.epochHistory[tag]
		if !ok || len(vals) == 0 {
			continue
		}
		series = append(series, plotSeries{Name: tag, Values: vals})
	}
	return series
}

// gatherTimingSeries collects timing history for the given tags.
func (g *Graph) gatherTimingSeries(tags []string) []plotSeries {
	if g.timingHistory == nil {
		return nil
	}

	if len(tags) == 0 {
		for tag := range g.timingHistory {
			tags = append(tags, tag)
		}
		sort.Strings(tags)
	} else {
		tags = g.expandGroups(tags)
	}

	series := make([]plotSeries, 0, len(tags))
	for _, tag := range tags {
		vals, ok := g.timingHistory[tag]
		if !ok || len(vals) == 0 {
			continue
		}
		series = append(series, plotSeries{Name: tag, Values: vals})
	}
	return series
}

// writeCSV writes series data to a CSV file.
func writeCSV(path string, series []plotSeries) error {
	var b strings.Builder

	// Header.
	b.WriteString("epoch")
	for _, s := range series {
		b.WriteByte(',')
		b.WriteString(s.Name)
	}
	b.WriteByte('\n')

	// Find max epoch count.
	maxLen := 0
	for _, s := range series {
		if len(s.Values) > maxLen {
			maxLen = len(s.Values)
		}
	}

	// Data rows.
	for i := range maxLen {
		fmt.Fprintf(&b, "%d", i+1)
		for _, s := range series {
			b.WriteByte(',')
			if i < len(s.Values) {
				fmt.Fprintf(&b, "%.8g", s.Values[i])
			}
		}
		b.WriteByte('\n')
	}

	p := filepath.Clean(path)
	return os.WriteFile(p, []byte(b.String()), 0600)
}

// plotTemplate is the self-contained HTML template for training curves.
// The /*DATA*/ placeholder is replaced with JSON series data.
const plotTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Training Curves</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Helvetica,Arial,sans-serif;background:#f5f6fa;padding:24px}
.container{background:#fff;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.08);padding:24px;max-width:960px;margin:0 auto}
h2{color:#2c3e50;margin-bottom:16px;font-size:18px}
canvas{width:100%;cursor:crosshair}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:14px}
.legend-item{display:flex;align-items:center;gap:6px;font-size:12px;color:#555}
.legend-color{width:12px;height:12px;border-radius:2px}
.tooltip{position:absolute;background:rgba(44,62,80,.9);color:#fff;padding:6px 10px;border-radius:4px;font-size:11px;pointer-events:none;display:none;white-space:nowrap}
</style>
</head>
<body>
<div class="container">
<h2>Training Curves</h2>
<canvas id="chart" height="400"></canvas>
<div class="legend" id="legend"></div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA=/*DATA*/;
const COLORS=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#34495e','#c0392b','#2980b9','#27ae60','#d35400'];
const canvas=document.getElementById('chart');
const ctx=canvas.getContext('2d');
const tooltip=document.getElementById('tooltip');
const legend=document.getElementById('legend');

// HiDPI support.
const dpr=window.devicePixelRatio||1;
function resize(){
  const rect=canvas.getBoundingClientRect();
  canvas.width=rect.width*dpr;
  canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  draw();
}

// Chart margins.
const M={top:20,right:20,bottom:36,left:60};

function draw(){
  const W=canvas.width/dpr,H=canvas.height/dpr;
  const pw=W-M.left-M.right,ph=H-M.top-M.bottom;
  ctx.clearRect(0,0,W,H);

  if(!DATA||DATA.length===0)return;

  // Compute bounds.
  let maxEp=0,minV=Infinity,maxV=-Infinity;
  DATA.forEach(s=>{
    maxEp=Math.max(maxEp,s.values.length);
    s.values.forEach(v=>{minV=Math.min(minV,v);maxV=Math.max(maxV,v)});
  });
  if(minV===maxV){minV-=1;maxV+=1}
  const pad=(maxV-minV)*0.05;
  minV-=pad;maxV+=pad;

  // Axis helpers.
  const xScale=i=>M.left+(i/(maxEp-1||1))*pw;
  const yScale=v=>M.top+ph-(v-minV)/(maxV-minV)*ph;

  // Grid lines.
  ctx.strokeStyle='#eee';ctx.lineWidth=1;
  const yTicks=5;
  for(let i=0;i<=yTicks;i++){
    const v=minV+(maxV-minV)*i/yTicks;
    const y=yScale(v);
    ctx.beginPath();ctx.moveTo(M.left,y);ctx.lineTo(W-M.right,y);ctx.stroke();
    ctx.fillStyle='#999';ctx.font='10px Helvetica';ctx.textAlign='right';
    ctx.fillText(formatVal(v),M.left-6,y+3);
  }
  // X-axis ticks.
  const xStep=Math.max(1,Math.floor(maxEp/10));
  ctx.textAlign='center';
  for(let i=0;i<maxEp;i+=xStep){
    const x=xScale(i);
    ctx.beginPath();ctx.moveTo(x,M.top);ctx.lineTo(x,H-M.bottom);ctx.stroke();
    ctx.fillStyle='#999';ctx.font='10px Helvetica';
    ctx.fillText(''+(i+1),x,H-M.bottom+14);
  }

  // Axis labels.
  ctx.fillStyle='#888';ctx.font='11px Helvetica';ctx.textAlign='center';
  ctx.fillText('Epoch',M.left+pw/2,H-4);

  // Draw lines.
  DATA.forEach((s,si)=>{
    const color=COLORS[si%COLORS.length];
    ctx.strokeStyle=color;ctx.lineWidth=2;
    ctx.beginPath();
    s.values.forEach((v,i)=>{
      const x=xScale(i),y=yScale(v);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
    // Points.
    s.values.forEach((v,i)=>{
      ctx.fillStyle=color;ctx.beginPath();
      ctx.arc(xScale(i),yScale(v),3,0,Math.PI*2);ctx.fill();
    });
  });

  // Store layout for hover.
  canvas._layout={xScale,yScale,maxEp,minV,maxV,pw,ph};
}

function formatVal(v){
  if(Math.abs(v)<0.001&&v!==0)return v.toExponential(1);
  if(Math.abs(v)>=1000)return v.toFixed(0);
  if(Math.abs(v)>=1)return v.toFixed(3);
  return v.toFixed(4);
}

// Legend.
DATA.forEach((s,i)=>{
  const item=document.createElement('div');item.className='legend-item';
  const swatch=document.createElement('div');swatch.className='legend-color';
  swatch.style.background=COLORS[i%COLORS.length];
  const label=document.createTextNode(s.name);
  item.appendChild(swatch);item.appendChild(label);
  legend.appendChild(item);
});

// Tooltip on hover.
canvas.addEventListener('mousemove',e=>{
  const L=canvas._layout;if(!L)return;
  const rect=canvas.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  // Find nearest epoch.
  let bestDist=Infinity,bestEp=-1;
  for(let i=0;i<L.maxEp;i++){
    const d=Math.abs(L.xScale(i)-mx);
    if(d<bestDist){bestDist=d;bestEp=i}
  }
  if(bestDist>20){tooltip.style.display='none';return}
  let html='<b>Epoch '+(bestEp+1)+'</b>';
  DATA.forEach((s,si)=>{
    if(bestEp<s.values.length){
      const c=COLORS[si%COLORS.length];
      html+='<br><span style="color:'+c+'">■</span> '+s.name+': '+formatVal(s.values[bestEp]);
    }
  });
  tooltip.innerHTML=html;tooltip.style.display='block';
  tooltip.style.left=(e.pageX+12)+'px';tooltip.style.top=(e.pageY-10)+'px';
});
canvas.addEventListener('mouseleave',()=>{tooltip.style.display='none'});

window.addEventListener('resize',resize);
resize();
</script>
</body>
</html>`
