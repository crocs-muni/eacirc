#Script sample for creation script for downloading results.
#All changes here will propagate to the resulting script.

#IN CASE OF CHANGING ONE OR MORE KEYWORDS, CHANGE CONSTANTS
#ACCORDINGLY IN FILE ONECLICKCONSTANTS.H. OTHERWISE ONECLICK
#APPLICATION WONT WORK. KEYWORDS ARE UPPERCASE AND END WITH _KW.

#Change this script if BOINC web interface or some URL changes.

#!/usr/bin/perl

use strict;
use warnings;
use WWW::Mechanize;
use Archive::Zip qw( :ERROR_CODES :CONSTANTS );
use File::Path;
use Term::ReadKey;

#Script default values constants
#SETTING PROJECT ID
#11: Main EACirc application
#3 : CUDA testing and debug
#14: EACirc testing and debug
use constant PROJECT_ID => 'PROJECT_ID_KW';
#Timeout limit, if you still get timeout error raise this number
use constant TIMEOUT => 600;
use constant RESULT_DIR => './results/';
use constant FILES_COUNT => '2';
use constant LOGIN_URL => 'http://centaur.fi.muni.cz:8000/boinc/labak_management';
use constant DIRZIP_URL => 'http://centaur.fi.muni.cz:8000/boinc/labak/dirZip?dir=';


sub login();
sub download_rem_dir($);
sub create_directory($);
sub extract_delete_archive($);

my $usr;
my $pwd;
my $mech = WWW::Mechanize->new(timeout => TIMEOUT);
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

    create_directory(RESULT_DIR);

    DOWNLOAD_REM_DIR_KW      ('REM_DIR_NAME_KW');
    EXTRACT_DELETE_ARCHIVE_KW('ARCHIVE_NAME_KW');
    #Leave this comment here.
    #There has to be at least one line beginning with four spaces (that's pretty specific) after method prototype.

    print "Press ENTER to exit.\n";
    <STDIN>;
}

#Performs check for local and remote files existence.
#If files exists locally or dont exist remotely, writes error, doesnt download.
#Otherwise downloads and stores archive with remote directory.
#Remote directory have to be in results directory of EACirc project on BOINC server.
sub download_rem_dir($) {
    my $dir = shift;
    my $file = RESULT_DIR . $dir . ".zip";
    
    #Download script adress URL + prefix
    my $prefix = DIRZIP_URL;
    #Download script suffix, additional arguments for dirZip.php are 
    #set here.
    my $suffix = '&count=' . FILES_COUNT . '&appid=' . PROJECT_ID;
    
    #Check for local file existence
    if(-e $file) {
        print "$file already exists, didn't download.\n";
        return;
    }
    my $url = $prefix . $dir . $suffix;
    eval {
        $mech->get("$url");
    };
    if($@) { 
        #If you get here try to set higher timeout limit
        print "Timeout recovery, $file wasn't downloaded!\n";
        $mech = WWW::Mechanize->new(timeout => TIMEOUT);
        login;
        return;
    } 
    
    #Check for existence of file on BOINC server.
    #If error occurs does nothing.
    if($mech->content() =~ /An error occurred/) {
        print "Error on server occurred when downloading directory $dir\n";
        return;
    }
    
    #Downloading the file
    $mech->save_content($file);
    print "$file downloaded.\n";
}

#Extracts given archive into ./results/ directory
#If given string is not filename does nothing.
#After succesfull extraction deletes archive.
#In case error occurs, writes error, file is not deleted.
#UNIX: Default folder permissions are 0777, umask is applied.
sub extract_delete_archive ($) {
    my $name = shift;
    $name = RESULT_DIR . $name;
    
    if (-e $name) {
        my $archive = Archive::Zip->new();
        unless ( $archive->read($name) == AZ_OK ) {
            die 'read error';
        }
        my @members = $archive->members();

        foreach my $element (@members) {
            if ($element->isDirectory()) {
                #This has to be done manually, Archive::Zip creates
                #directories with default permissions 0666 (...)
                mkpath(RESULT_DIR . $element->fileName());
            } else {
                unless ($archive->extractMember($element , RESULT_DIR . $element->fileName()) == AZ_OK) {
                    die 'read error';
                }
            }
        }
        #Delete downloaded archive
        unlink $name;
    }
}

#Creates folder if nonexistent, otherwise does nothing
#UNIX: Default perrmissions are 0777, umask is applied
sub create_directory ($) {
    my $directory = shift;
    if(-e $directory) {
        return;
    }
    mkdir $directory;
}

#Logs into BOINC.
#If login is unsuccesfull, terminates script.
sub login () {
    #LOGIN PAGE URL
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
