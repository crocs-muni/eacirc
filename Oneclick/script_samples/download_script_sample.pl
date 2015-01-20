#Script sample for creation script for downloading results.
#All changes here will propagate to the resulting script.

#IN CASE OF CHANGING ONE OR MORE KEYWORDS, CHANGE CONSTANTS
#ACCORDINGLY IN FILE ONECLICKCONSTANTS.H. OTHERWISE ONECLICK
#APPLICATION WONT WORK. KEYWORDS ARE UPPERCASE.

#Change this script if BOINC web interface or some URL changes.

#!/usr/bin/perl

use strict;
use warnings;
use WWW::Mechanize;
sub login($$$);
sub download_clones($$$);
sub download_file($$$);
sub create_directory($);

{
	my $mech = WWW::Mechanize->new(autocheck => 1);
	#Enter login data here
	#print q(Name: );
	my $usr = 'username';
	#print q(Pwd : );
	my $pwd = 'password';
	
	login($usr , $pwd , $mech);
	create_directory('results');
	
	CREATE_DIRECTORY('DIRECTORY_PATH');
	DOWNLOAD_CLONES('WU_NAME' , 'WU_DIRECTORY' , $mech);
	#There has to be at least one line beginning with \t after method prototype. Leave this comment here.
	
	print "If all workunit results were not downloaded, wait for their completion and run this script again.\n";
	print "Press ENTER to exit.\n";
	<STDIN>;
}

#Logs into BOINC.
#If login is unsuccesfull, terminates script.
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

#Downloads results of all clones of one workunit.
#If you wish to download additional files from BOINC specify it here.
sub download_clones ($$$) {
	my ($wu , $directory , $mech) = (shift , shift , shift);
	#URL WITH FILE FETCHER ON BOINC
	my $fetcher_url = 'http://centaur.fi.muni.cz:8000/boinc/labak/fetcher.php?filename=';
	
	#This loop will download all clones of one workunit.
	#Originally, config and log file is downloaded for each workunit.
	#If you wish to change that, add corresponding line into the loop.
	for (my $i = 1 ; $i <= CLONE_COUNT ; $i++) {
		download_file("$fetcher_url$wu"."[$i]_0_0" , "$directory"."config_$i.xml" , $mech);
		download_file("$fetcher_url$wu"."[$i]_0_1" , "$directory"."log_$i.log" , $mech);
	}
}

#Performs check for local and remote files existence.
#If files exists locally or dont exist remotely, writes error, doesnt download.
#Otherwise downloads and stores files.
sub download_file ($$$) {
	my ($url , $file , $mech) = (shift , shift , shift);
	
	#Check for local file existence
	if(-e $file) {
		print "$file already exists, didn't download.\n";
		return;
	}
	
	$mech->get("$url");
	
	#Check for existence of file on BOINC server.
	#If nonexistent does nothing.
	if($mech->content() =~ /An error occurred/) {
		print "File $url is not on BOINC server. Try again later or check filename.\n";
		return;
	}
	
	#Downloading the file
	$mech->save_content($file);
}

#Creates folder if nonexistent, otherwise does nothing
sub create_directory ($) {
	my ($directory) = (shift);
	if(-e $directory) {
		return;
	}
	mkdir $directory;
}

