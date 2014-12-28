#Script sample for uploading work to BOINC.
#All changes here will propagate to the resulting script.

#IN CASE OF CHANGING ONE OR MORE KEYWORDS, CHANGE ACCORDING
#CONSTANTS IN FILE ONECLICKCONSTANTS.H. OTHERWISE ONECLICK
#APPLICATION WONT WORK. KEYWORDS ARE UPPERCASE.

#Change this script if BOINC web interface or some URL changes.

#!/usr/bin/perl
use warnings;
use strict;
use WWW::Mechanize;
sub create_wu ($$$);
sub login ($$$);

#Main method
{
	my $mech = WWW::Mechanize->new(autocheck => 1);
	#Enter login data here
	#print q(Name: );
	my $usr = q(username);
	#print q(Pwd : );
	my $pwd = q(password);
	
	login($usr , $pwd , $mech);
	
	CREATE_WU('WU_NAME' , 'CONFIG_PATH' , $mech);
	#There has to be at least one line beginning with \t after method prototype. Leave this comment here.
	
	print "Press ENTER to exit.\n";
	<STDIN>;
}

#This subroutine will log you in
sub login ($$$) {
	my($usr , $pwd , $agent) = (shift , shift , shift);
	#LOGIN PAGE URL
	my $url = 'http://centaur.fi.muni.cz:8000/boinc/labak_management';
	$agent->get($url);
	#Login to boinc web interface 
	$agent->form_number(1);
	$agent->field('login' , $usr);
	$agent->field('password' , $pwd);
	$agent->click('submit');
	#Successful login check 
	if($agent->content() =~ /errors/) {
		print "Invalid name/password combination.\n";
		print "Press ENTER to exit.\n";
		<STDIN>;
		exit;
	}
}

#This subroutine will create single workunit
sub create_wu ($$$) {
	my($wu_name , $config_path , $mech) = (shift , shift , shift);
	#Creating single workunit
	#CREATE WORK URL
	my $url = 'http://centaur.fi.muni.cz:8000/boinc/labak_management/work/create';
	$mech->get($url);
	#Step 1/4
	$mech->form_number(1);
	#SETTING PROJECT ID, 11 is default EACirc project
	$mech->select('step0[appid]' , '11');
	$mech->field('step0[name]' , $wu_name);
	$mech->click('next-step');
	#Unique WU name check
	if($mech->content =~ /errors/) { 
		print "WU name already in use. Try manual renaming or deleting old WU.\n";
		return;
	}
	#Step 2/4
	$mech->form_number(1);
	$mech->field('infiles_upload[]' , $config_path , 1);
	$mech->click('next-step');
	#Step 3/4
	$mech->form_number(1);
	$mech->field('wu_batch' , 'CLONE_COUNT');
	$mech->click('next-step');
	#Step 4/4
	$mech->form_number(1);
	$mech->field('delay_bound' , 'DELAY_BOUND');
	$mech->click('next-step');
	#Creation done.
}


