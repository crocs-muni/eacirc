#include "FileGenerator.h"

FileGenerator::FileGenerator(std::string path) {
	parser = new ConfigParser(path);
	generateFiles();
}

FileGenerator::~FileGenerator() {
	delete parser;
}

void FileGenerator::generateFiles() {
	std::vector<vector<int>> algorithmsRounds = parser->getAlgorithmsRounds();
	std::vector<int> numGenerations = parser->getNumGenerations();
	TiXmlNode * root = parser->getRoot();
	TiXmlNode * eacNode = NULL;
	eacNode = getXMLElement(root , PATH_EACIRC);
	int project = parser->getProject();
	std::string wuIdentifier = parser->getWuIdentifier();
	std::ofstream upScript;
	upScript.open(UPLOAD_SCRIPT , ios::out);
	if(!upScript.is_open()) throw runtime_error("can't open output file for upload script: " + (string)UPLOAD_SCRIPT);

	writeFirstUpScript(upScript);

	for(int i = 0 ; i < numGenerations.size() ; i++) {
		for(int k = 0 ; k < algorithmsRounds.size() ; k++) {
			for(int l = 1 ; l < algorithmsRounds[k].size() ; l++) {
				std::string configName;
				std::string notes;
				TiXmlNode * n = NULL;
				std::string projectName;
				std::string algorithmName;
				//Here, tags in config file are set and human readable description of alg and project are given.
				setAlgorithmSpecifics(root , project , algorithmsRounds[k][0] , &projectName , &algorithmName);
				notes = algorithmName;

				configName = (wuIdentifier + "_" + projectName + "_" + itostr(algorithmsRounds[k][0]) + "_" + algorithmName +
					+ "_r" + itostr(algorithmsRounds[k][l]) + "_" + itostr(numGenerations[i]) + "-gen" + ".xml");

				upScript << "create_wu(q(" << configName << ") , q("<< PATH_CFGS_FOLDER << configName << ") , $mech);" << std::endl;

				notes.append(" function with " + itostr(algorithmsRounds[k][l]) + " rounds.");
				setXMLElementValue(root , PATH_EAC_NOTES , notes);
				setXMLElementValue(root , PATH_EAC_GENS , itostr(numGenerations[i]));
				n = eacNode->Clone();
				
				if(saveXMLFile(n , PATH_CFGS_FOLDER + configName) != STAT_OK) 
					throw runtime_error("can't save file (folder must exist): " + (string)PATH_CFGS_FOLDER + configName);

				notes.clear();
				configName.clear();
				algorithmName.clear();
				projectName.clear();
			}
		}
	}
	writeSecondUpScript(upScript);
	upScript.close();
}

void FileGenerator::writeFirstUpScript(std::ofstream & script) {
	script << "#!/usr/bin/perl" << std::endl;
	script << "use warnings;" << std::endl;
	script << "use strict;" << std::endl;
	script << "use WWW::Mechanize;" << std::endl;
	script << "sub create_wu ($$$);" << std::endl;
	script << "sub login ($$$);" << std::endl;
	//Main method
	script << "{my $mech = WWW::Mechanize->new(autocheck => 1);" << std::endl;
	//Login... enter your credentials here
	script << "my $usr = q(username);" << std::endl;
	script << "my $pwd = q(password);" << std::endl;
	script << "login($usr , $pwd , $mech);" << std::endl;
}

void FileGenerator::writeSecondUpScript(std::ofstream & script){
	script << "}" << std::endl;
	//Definition of subroutines - login to server
	script << "sub login ($$$) {" << std::endl;
	script << "my($usr , $pwd , $agent) = (shift , shift , shift);" << std::endl;
	script << "my $url = q(http://centaur.fi.muni.cz:8000/boinc/labak_management/);" << std::endl;
	script << "$agent->get($url);" << std::endl;
	script << "$agent->form_number(1);" << std::endl;
	script << "$agent->field(q(login) , $usr);" << std::endl;
	script << "$agent->field(q(password) , $pwd);" << std::endl;
	script << "$agent->click(q(submit));" << std::endl;
	script << "#Successful login check " << std::endl;
	script << "if($agent->content() =~ /errors/) {" << std::endl;
	script << "print qq(Invalid name/password combination.\n);" << std::endl;
	script << "exit;}}" << std::endl;
	script << "sub create_wu ($$$) {" << std::endl;
	script << "my($wu_name , $config_path , $mech) = (shift , shift , shift);" << std::endl;
	//Creation of single workunit
	script << "my $url = qq(http://centaur.fi.muni.cz:8000/boinc/labak_management/work/create);" << std::endl;
	script << "$mech->get($url);" << std::endl;
	script << "$mech->form_number(1);" << std::endl;
	script << "$mech->select(q(step0[appid]) , q(11));" << std::endl;
	script << "$mech->field(q(step0[name]) , $wu_name);" << std::endl;
	script << "$mech->click(q(next-step));" << std::endl;
	script << "if($mech->content =~ /errors/) { " << std::endl;
	script << "print qq(WU name already in use. Try manual renaming or deleting old WU.\n);" << std::endl;
	script << "return;}" << std::endl;
	script << "$mech->form_number(1);" << std::endl;
	script << "$mech->field(q(infiles_upload[]) , $config_path , 1);" << std::endl;
	script << "$mech->click(q(next-step));" << std::endl;
	script << "$mech->form_number(1);" << std::endl;
	script << "$mech->field(q(wu_batch) , q("<< parser->getClones() <<"));" << std::endl;
	script << "$mech->click(q(next-step));" << std::endl;
	script << "$mech->form_number(1);" << std::endl;
	script << "$mech->field(q(delay_bound) , q(" << parser->getDelayBound() << "));" << std::endl;
	script << "$mech->click(q(next-step));}" << std::endl;
}