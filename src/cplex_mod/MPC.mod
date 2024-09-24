/*********************************************
 * OPL 12.10.0.0 Model
 * Author: yangk
 * Creation Date: Aug 25, 2020 at 10:45:21 AM
 *********************************************/
tuple demandAttrTuple{
  	int i;
  	int j;
  	int t;
  	float v;
  	float tt;
  	float p;
}

tuple edgeAttrTuple{
  	int i;
  	int j;
  	int t;
}

tuple Edge{
  int i;
  int j;
}

tuple demandEdgeTuple{
  int i;
  int j;
  int t;
}

tuple accTuple{
  int i;
  float n;
}

tuple daccTuple{
  int i;
  int t;
  float n;
}

string path = ...;
int t0 = ...;
int T = ...;
int tf = t0+T;
float beta = ...;
{demandAttrTuple} demandAttr = ...;
{edgeAttrTuple} edgeAttr = ...;
{accTuple} accInitTuple = ...;
{daccTuple} daccAttr = ...;

{Edge} edge = {<i,j>|<i,j,t> in edgeAttr};
{int} region = {i|<i,v> in accInitTuple};
float accInit[region] = [i:v|<i,v> in accInitTuple];
float dacc[region][t0..tf-1] = [i:[t:v]|<i,t,v> in daccAttr];
{demandEdgeTuple} demandEdge = {<i,j,t>|<i,j,t,v,tt,p> in demandAttr};
float demand[demandEdge] = [<i,j,t>:v|<i,j,t,v,tt,p> in demandAttr];
float price[demandEdge] = [<i,j,t>:p|<i,j,t,v,tt,p> in demandAttr];
float demandTime[demandEdge] = [<i,j,t>:tt|<i,j,t,v,tt,p> in demandAttr];
int tt[edge] = [<i,j>:t|<i,j,t> in edgeAttr];
dvar float+ demandFlow[edge][t0..tf-1];
dvar float+ rebFlow[edge][t0..tf-1];
dvar float+ acc[region][t0..tf];
maximize(sum(e in demandEdge) demandFlow[<e.i,e.j>][e.t]*price[e] - beta * sum(e in edge,t in t0..tf-1)rebFlow[e][t]*tt[e]  - beta * sum(e in edge,t in t0..tf-1:<e.i,e.j,t> in demandEdge)demandFlow[e][t]*demandTime[<e.i,e.j,t>]);
subject to
{
  forall(t in t0..tf-1)
  {
    forall(i in region)
    {  
    	acc[i][t+1] == acc[i][t] - sum(e in edge: e.i==i)(demandFlow[e][t] + rebFlow[e][t]) 
      			+ sum(e in demandEdge: e.j==i && e.t+demandTime[e]==t)demandFlow[<e.i,e.j>][e.t] + sum(e in edge: e.j==i && t-tt[e]>=t0)rebFlow[e][t-tt[e]] + dacc[i][t];
		sum(e in edge: e.i==i)(demandFlow[e][t]+ rebFlow[e][t]) <= acc[i][t];
      	if(t == t0)
      		acc[i][t] == accInit[i];
 	}  	    
    forall(e in edge)
      if(<e.i,e.j,t> in demandEdge)
      		demandFlow[e][t] <= demand[<e.i,e.j,t>];
      else
      		demandFlow[e][t] == 0;      
  }
  
}

main {
  thisOplModel.generate();
  cplex.solve();
  var t = thisOplModel.t0
  var ofile = new IloOplOutputFile(thisOplModel.path);  

  ofile.write("flow=[")
  for(var e in thisOplModel.edge)
	if(thisOplModel.demandFlow[e][t]>1e-3 || thisOplModel.rebFlow[e][t]>1e-3)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(thisOplModel.demandFlow[e][t]);
		 ofile.write(",");
		 ofile.write(thisOplModel.rebFlow[e][t]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.close();
}




