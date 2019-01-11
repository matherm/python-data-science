
RAW = "aCAunecsLfnsetAneoilasmpoaBdaethkCbMnCtDieeAntMrdDBFcResEelodytlaeuCubTdeGTMaolweQtnOnErJhNATsEMSMUurorltrThBeVacdiluesDmkeaoknetroJhharai5esboceasTiusteiFdMiShtptarrtaungKcuntareltdorhiHhlsinMprSOWEgtusradonBaoSaaSaeehesoFlkMtchhMbWDhaoaeohoDJAmeudngo"
R_LIST = list(RAW)

BANDS = ["The Libertines","Die Antwoord","The Vaccines","Deadmau5","Death Cab For Cutie","Of Monsters and Men","SOHN","Eagulls","Kasabian"]

for band in BANDS:
	for c in band:
		hit = 0;
		for i in range(len(R_LIST)):
			if R_LIST[i] == c:
				R_LIST[i] = "_"
				hit = 1
				break;
		if hit == 0:
			if not c == " ":
				print str("fail: " + c + " in " + band)

RAW_TEMP = "".join(R_LIST)
print RAW_TEMP

