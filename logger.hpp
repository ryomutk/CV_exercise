#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

class logger
{
private:
    string path;
    stringstream content;
    string headers;
    int countCommaInLine();
    void replaceDataInLine(int index);

public:
    logger(string path, string columns);
    void append(string data);
    void nextLine();
    void writeFile();
};

#endif // LOGGER_HPP