%Custom implementation. May not work for all cases
%Take care of Inf and Nan entries 

filelist = readdir("../dataset/ASVspoof2017_train_dev/wav/train/");
numel(filelist)
for ii = 1:numel(filelist)
	if(regexp(filelist{ii},"^\\.\\.?$"))
		continue;
	endif
	[audio,fs] = audioread(["../dataset/ASVspoof2017_train_dev/wav/train/" filelist{ii}]);
	[ceps] = mfcc(audio,fs);
	label = 0;
	filelist{ii}
	if(ii>=1511)
		label = 1
	endif
	features = [];
	for columns=1:size(ceps)(2)
		if(any(isnan(ceps(:,columns)))==1)
			continue;
		endif
		features = [features [ceps(:,columns);0] ];
		%csvwrite(["features/" filelist{ii} num2str(columns) "features.csv"], [ceps(:,columns);label]');
	endfor
	csvwrite(["features/" filelist{ii} "features.csv"], features');
endfor
