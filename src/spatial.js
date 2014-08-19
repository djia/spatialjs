/**
 * JavaScript Spatial Statistics Package
 * Author: David Wei Jia
 * Copyright (c) 2014 David Wei Jia
 * Stanford University
 */

var Spatial = {};

// define some constants
Spatial.PHI = (1 + Math.sqrt(5)) / 2;
Spatial.RES_PHI = 2 - Spatial.PHI;
Spatial.TAU_DEFAULT = 0.001;

Spatial.trace = function(M) {
	var n = M.length;
	var traceVal = 0;
	for(var i = 0; i < n; i++) {
		traceVal += M[i][i];
	}
	return traceVal;
}

Spatial.normVec = function(vec) {
	return Math.sqrt(Spatial.normVecSqr(vec));
}

Spatial.normVecSqr = function(vec) {
	var norm = 0;
	for(var i = 0; i < vec.length; i++) {
		norm += typeof vec[i] == 'number' ? Math.pow(vec[i], 2) : Math.pow(vec[i][0], 2);
	}
	return norm;
}

// calculate the trace of the square of a matrix
// M must be n x n
Spatial.traceMSqr = function(M) {
	var n = M.length;
	var traceVal = 0;
	for(var i = 0; i < n; i++) {
		var row = M[i];
		for(var j = 0; j < n; j++) {
			traceVal += row[j] * M[j][i];
		}
	}
	return traceVal;
}

// calculate the trace of the square of a matrix
// M must be symmetric n x n
Spatial.traceMSqrSymmetric = function(M) {
	var n = M.length;
	var traceVal = 0;
	for(var i = 0; i < n; i++) {
		traceVal += Spatial.normVecSqr(M[i]);
	}
	return traceVal;
}

Spatial.goldenSectionSearch = function(a, b, c, tau, func) {
	var x;
	if(c - b > b - a) {
		x = b + Spatial.RES_PHI * (c - b);
	} else {
		x = b - Spatial.RES_PHI * (b - a);
	}
	if(Math.abs(c - a) <= tau * (Math.abs(b) + Math.abs(x))) {
		return (c + a) / 2;
	}
	if(func(x) < func(b)) {
		if(c - b > b - a) {
			return Spatial.goldenSectionSearch(b, x, c, tau, func);
		} else {
			return Spatial.goldenSectionSearch(a, x, b, tau, func);
		}
	} else {
		if(c - b > b - a) {
			return Spatial.goldenSectionSearch(a, b, x, tau, func);
		} else {
			return Spatial.goldenSectionSearch(x, b, c, tau, func);
		}
	}
}

Spatial.SAR = function(y, X, W, options) {
	// check that y and X are in the right format
	if(typeof y[0] == 'number') {
		var yTemp = y;
		y = [];
		for(var i = 0; i < yTemp.length; i++) {
			y.push([yTemp[i]]);
		}
	}
	
	if(typeof X[0] == 'number') {
		var XTemp = X;
		X = [];
		for(var i = 0; i < XTemp.length; i++) {
			X.push([XTemp[i]]);
		}
	}
	
	var op = options ? options : {};
	
	if(op.verbose) console.log('Got in approxSAR');
	
	var n = y.length;
	
	// store IX which is constant for each iteration
	if(op.verbose) var startTime = Date.now();
	var XPseudo = numeric.dot(numeric.inv(numeric.dot(numeric.transpose(X), X)), numeric.transpose(X));
	var IX = numeric.sub(numeric.identity(X.length), numeric.dot(X, XPseudo));
	if(op.verbose) var endTime = Date.now();
	if(op.verbose) console.log("Finished computing IX: " + (endTime - startTime));
	
	if(op.verbose) var startTime = Date.now();
	
	// B = IX * y
	var B = numeric.dot(IX, y);
	// IXSar1 = ||IX * y||^2
	var IXSqr1 = Spatial.normVecSqr(B);
	
	// C = IX * W * y
	var C = numeric.dot(IX, numeric.dot(W, y));
	// IXSqr2 = IXSqr3 = y^T * IX^T IX * W * y
	var IXSqr2 = numeric.sum(numeric.mul(C, B));
	// IXSqr4 = ||IX * W * y||^2
	var IXSqr4 = Spatial.normVecSqr(C);
	
	if(op.verbose) var endTime = Date.now();
	if(op.verbose) console.log("Finished computing IXSqr: " + (endTime - startTime));
	
	// Start Chebychev approximation for ln|A|
	var td1 = 0;
	
	if(op.verbose) var startTime = Date.now();
	// var WSqr = numeric.dot(W, W);
	var td2 = Spatial.traceMSqr(W);
	if(op.verbose) var endTime = Date.now();
	if(op.verbose) console.log("Finished computing trace(WSqr): " + (endTime - startTime));
	
	var chebyPolyCoeffs = [[1, 0, 0], [0, 1, 0], [-1, 0, 2]];
	var nposs = 3;
	var seqlnposs = [[1], [2], [3]];
	var tdvec = [[n], [td1], [td2 - 0.5 * n]];
	
	var xk = [[0], [0], [0]];
	for(var k = 0; k < nposs; k++) {
		xk[k][0] = Math.cos(Math.PI * (seqlnposs[k][0] - 0.5) / nposs);
	}
	
	// update the logDet variable each time so that the last iteration
	// will yield ln|Aopt|, will use this later to calculate the approximate |A|
	// for ML value calculation
	var logDet;
	
	var func = function(rho) {
		if(op.verbose) console.log("Got in rho estimation func");
		
		var SSE = IXSqr1 - 2 * rho * IXSqr2 + Math.pow(rho, 2) * IXSqr4;
		if(op.verbose) console.log("SSE: " + SSE);
		
		var logSSE = Math.log(SSE);
		if(op.verbose) console.log("logSSE: " + logSSE);
		
		// cheby approximation
		var cposs = [[0, 0, 0]];
		for(var j = 0; j < nposs; j++) {
			var temp = 0;
			for(var k = 0; k < nposs; k++) {
				temp += (2.0 / nposs) * Math.log(1.0 - (rho + 0.0) * xk[k][0]) * Math.cos(Math.PI * j * (seqlnposs[k][0] - 0.5) / nposs);
			}
			cposs[0][j] = temp;
		}

		var chebyLogDetApprox = numeric.dot(numeric.dot(cposs, chebyPolyCoeffs), tdvec);
		logDet = chebyLogDetApprox[0][0];
		if(op.verbose) console.log("logDet: " + logDet);
		
		var val = (-2 / n) * logDet + logSSE;
		if(op.verbose) console.log('Func Val: ' + val);
		
		return val;
	}
	
	var tau = op.tau ? op.tau : Spatial.TAU_DEFAULT;
	var a = -1;
	var c = 1;
	var b = (c + Spatial.PHI * a) / (1 + Spatial.PHI);
	
	var rhoOpt = Spatial.goldenSectionSearch(a, b, c, tau, func);
	
	var AyOpt = numeric.sub(y, numeric.mul(rhoOpt, numeric.dot(W, y)));
	var beta = numeric.dot(XPseudo, AyOpt);
	var sigmaSqr = (IXSqr1 - 2 * rhoOpt * IXSqr2 + Math.pow(rhoOpt, 2) * IXSqr4) / n;
	
	// ML calculation
	var logML = -(n / 2) * Math.log(2 * Math.PI * sigmaSqr) - 1 / (2 * sigmaSqr) * Spatial.normVecSqr(numeric.sub(AyOpt, numeric.dot(X, beta))) + logDet;
	
	var params = {
		rho: rhoOpt,
		beta: beta,
		sigmaSqr: sigmaSqr,
		logML: logML
	}
	
	if(op.verbose) console.log(params);
	
	return params;
}