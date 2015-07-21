#Script sample for uploading work to BOINC.
#All changes here will propagate to the resulting script.

#IN CASE OF CHANGING ONE OR MORE KEYWORDS, CHANGE CONSTANTS
#ACCORDINGLY IN FILE ONECLICKCONSTANTS.H. OTHERWISE ONECLICK
#APPLICATION WONT WORK. KEYWORDS ARE UPPERCASE AND END WITH _KW.

#Change this script if BOINC web interface or some URL changes.

#!/usr/bin/perl
use warnings;
use strict;
use WWW::Mechanize;
use HTTP::Cookies;
use Term::ReadKey;

#Script constants
#SETTING PROJECT ID
#11: Main EACirc application
#3 : CUDA testing and debug
#14: EACirc testing and debug
use constant PROJECT_ID => 'PROJECT_ID_KW';
#Timeout limit on batch creation, 1 is minimum
#Higher number will eliminate overlapping job creation but it will take longer
use constant TIMEOUT => 1;
use constant CLONES => 'CLONES_KW';
use constant LOGIN_URL => 'http://centaur.fi.muni.cz:8000/boinc/labak_management';
use constant CREATE_WORK_URL => 'http://centaur.fi.muni.cz:8000/boinc/labak_management/work/create';

sub create_wu ($$);
sub login();

#Global variables declaration
my $cookie_jar = HTTP::Cookies->new();
my $mech = WWW::Mechanize->new(cookie_jar => $cookie_jar);
my $usr;
my $pwd;

#Main method
{
    #Enter login data here
    print "Signing in to url " . LOGIN_URL . "\n";
    print 'Name: ';
    chomp($usr = <STDIN>);
    print 'Pwd : ';
    ReadMode('noecho');
    chomp($pwd = <STDIN>);
    ReadMode(0);
    print "\n";
    login;

    CREATE_WU_KW('WU_NAME_KW' , 'CONFIG_PATH_KW');
    #Leave this comment here.
    #There has to be at least one line beginning with four spaces (that's pretty specific) after method prototype.

    print "Press ENTER to exit.\n";
    <STDIN>;
}

#This subroutine will log you in
sub login () {
    my $url = LOGIN_URL;
    $mech->get($url);
    #Login to boinc web interface 
    $mech->form_number(1);
    $mech->field('login' , $usr);
    $mech->field('password' , $pwd);
    $mech->click('submit');
    #Successful login check 
    if($mech->content() =~ /errors/) {
        print "Invalid name/password combination.\n";
        print "Press ENTER to exit.\n";
        <STDIN>;
        exit;
    }
}

#This subroutine will create single workunit
sub create_wu ($$) {
    my($wu_name , $config_path) = (shift , shift);
    #Creating single workunit
    my $url = CREATE_WORK_URL;
    $mech->get($url);
    #Step 1/4
    $mech->form_number(1);
    $mech->select('step0[appid]' , PROJECT_ID);
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
    $mech->field('wu_batch' , CLONES);
    $mech->click('next-step');
    #Step 4/4
    #Default values
    #If you wish to change some values, uncomment and proceed!
    $mech->form_number(1);
    #$mech->field('command_line' , '-c FILE_0');
    #$mech->field('clone_limit' , '0');
    #$mech->field('delay_bound' , '288000');
    #$mech->field('min_quorum' , '1');
    #$mech->field('target_nresults' , '1');
    #$mech->field('outfiles' , 'config.xml, eacirc.log, [scores.log], [fitness_progress.txt], [histograms.txt], [population_initial.xml], [population.xml], [state_initial.xml], [state.xml], [EAC_circuit.xml], [EAC_circuit.dot], [EAC_circuit.c], [EAC_circuit.txt]');
    
    $mech->timeout(TIMEOUT);
    eval {
        $mech->click('next-step');
    };
    if ($@) {
        $cookie_jar->clear;
        $mech = WWW::Mechanize->new(cookie_jar => $cookie_jar);
        login;
        #print "Timeout handled\n";
    }
    #Creation done.
    print "$wu_name created.\n";
}


