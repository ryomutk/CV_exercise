#include "logger.hpp"

logger::logger(string path, string columns)
{
    this->path = path;
    this->headers = columns;
}

void logger::append(string data)
{
    this->content << data << ",";
}

void logger::nextLine()
{
    this->content << " ";
}

void logger::writeFile()
{
    ofstream ofs(this->path);
    if (!ofs)
    {
        ofs.open(this->path, ios::out);
        ofs << headers;
        ofs << endl;
    }

    string line;
    while (!this->content.eof())
    {
        this->content >> line;
        ofs << line << endl;
    }

    ofs.close();
}
