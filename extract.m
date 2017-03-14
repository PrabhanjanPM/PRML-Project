%Custom implementation. May not work for all cases
%Take care of Inf and Nan entries 

filelist = readdir("../dataset/ASVspoof2017_train_dev/wav/train/");
numel(filelist)
for ii = 1:numel(filelist)
	if(regexp(filelist{ii},"^\\.\\.?$"))
		continue;
	endif
	filelist{ii}
	[audio,fs] = audioread(["../dataset/ASVspoof2017_train_dev/wav/train/" filelist{ii}]);
	[ceps] = mfcc(audio,fs);
	label = 0;
	if(ii>=1511)
		label = 1
	endif
	save(["features/" filelist{ii} "cepstral_features"],"ceps","label") 
	csvwrite(["features/" filelist{ii} "features.csv"], ceps);
endfor
