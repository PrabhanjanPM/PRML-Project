%Generate data using a Gaussian Mixture Model 
%Right now supports only 2 gaussian windows 

function gen_data(n,weight1,weight2,mean1,mean2,C1,C2)
	covariance1 = diag(C1)
	inv_covariance1 = diag(1./C1)


	covariance2 = diag(C2)
	inv_covariance2 = diag(1./C2)
	
	weight = weight1+weight2;
	X1 = [];
	X2 = [];
	X =  [];

	for i=1:100
		x1 = (stdnormal_rnd(1,n)*covariance1) + mean1;
		X1 = [X1;x1];
		x2 = (stdnormal_rnd(1,n)*covariance2) + mean2;
		X2 = [X2;x2];

		x = ((weight1/weight)*x1 + (weight2/weight)*x2);
		X = [X;x];
	endfor

	csvwrite("data1",X1);
	csvwrite("data2",X2);
	csvwrite("data" ,X);

endfunction 

%Usage
%weight1 = 0.5;
%weight2 = 0.5;
%mean1 = [-3 -3];
%mean2 = [3 3];
%C1 = [1 1];
%C2 = [1 1];
%gen_data(2,weight1,weight2,mean1,mean2,C1,C2);

