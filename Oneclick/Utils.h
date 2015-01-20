#ifndef UTILS_H
#define UTILS_H

#include <istream>
#include <string>
#include <fstream>
#include <sstream>
#include <exception>
#include <ctime>

//File with global methods declared

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

static class Utils {
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
	};

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
		//oneclickLogger << FileLogger::LOG_INFO << "created file " << path;
	};

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
	};

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
	};
};
#endif //UTILS_H