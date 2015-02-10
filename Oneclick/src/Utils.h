#ifndef UTILS_H
#define UTILS_H

#include <istream>
#include <string>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <ctime>
#include <vector>


#if defined WIN32 || defined _WIN32
#include <direct.h>
#elif defined __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#error "not implemeted for this platform"
#endif

//File with global methods declared

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

class Utils {
public:
	/** Converts integral value to string
	  * @param x		integer to be converted
	  * @return	string	converted integer
	  */
	static std::string itostr(int x) {
		std::stringstream ss;
		ss << x;
		return ss.str();
	};

	/** Opens file, reads it into string, closes file, returns string
	  * @param path				path to file
	  * @return					content of the file
	  * @throws runtime_error	when file can't be opened
	  */
	static std::string readFileToString(std::string path) {
		//oneclickLogger << FileLogger::LOG_INFO << "reading file " << path << " to string";
		std::ifstream file(path , std::ios::in);
		if(!file.is_open()) throw std::runtime_error("can't open input file: " + path);
		std::stringstream buffer;
		buffer << file.rdbuf();
		file.close();
		if(file.is_open()) throw std::runtime_error("can't close input file: " + path);
		return buffer.str();
    }

	/** Opens file, loads string into it, closes it.
	  * Source's content is erased.
	  * @param path				path to file
	  * @param source			string to be saved
	  * @throws runtime_error	when file can't be opened
	  */
	static void saveStringToFile(std::string path , std::string * source) {
		std::ofstream file(path , std::ios::out);
		if(!file.is_open()) throw std::runtime_error("can't open output file: " + path);
		file << *source;
		file.close();
		if(file.is_open()) throw std::runtime_error("can't close output file: " + path);
		source->clear();
    }

	/** Returns string after last separator in path.
	  * If no separator is found, whole path is returned.
	  * ../../example returns example
	  * @param path				path to be parsed
	  * @return					extracted directory or file name
	  */
	static std::string getLastItemInPath(std::string path) {
		std::string result;
		for(int i = path.length() - 1 ; i >= 0 ; i--) {
			if(path[i] == '/') {
				break;
			} else {
				result.insert(result.begin() , path[i]);
			}
		}
		return result;
	}

	/** Returns time in [hh:mm:ss] format.
	  * @return				time
	  */
	static std::string getTime() {
		time_t rawtime;
		struct tm * timeinfo;
		char buffer[80];
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(buffer , 80 , "%H:%M:%S" , timeinfo);
		std::stringstream temp;
		temp << "[" << buffer << "] ";
		return temp.str();
    }

	/** Retuns date in yy-mm-dd format.
	  * @return				date
	  */
	static std::string getDate() {
		time_t rawtime;
		struct tm * timeinfo;
		char buffer[80];
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(buffer , 80 , "%Y-%m-%d" , timeinfo);
		return std::string(buffer);
    }

	/** Splits string into shorter strings, separated by separator
	  * @param				toSplit string to be splitted
	  * @return				vector of strings
	  */
	static std::vector<std::string> split(std::string toSplit , char separator) {
		std::vector<std::string> result;
		std::string temp;
		for(int i = 0 ; i < toSplit.length() ; i++) {
			if(toSplit[i] != separator) {
				temp.push_back(toSplit[i]);
			} else {
				if(temp.length() > 0) result.push_back(temp);
				temp.clear();
			}
		}
		if(temp.length() > 0) result.push_back(temp);
		return result;
    }

	/** Creates directory.
	  * @param path			absolute or relative
	  */
	static void createDirectory(std::string path) {
		#if defined WIN32 || defined _WIN32
		if(_mkdir(path.c_str()) != 0) {
			if(errno != EEXIST) { throw std::runtime_error("error when creating directory: " + path); }
		}
		#elif defined __linux__
		if(mkdir(path.c_str() , 0777) != 0) {
			if(errno != EEXIST) { throw std::runtime_error("error when creating directory: " + path); }
		}
		#else
		throw std::runtime_error("not implemented for this platform");
		#endif
    }
};
#endif //UTILS_H
