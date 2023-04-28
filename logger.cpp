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
    ofstream ofs;
    if (!filesystem::exists(this->path))
    {
        ofs.open(this->path);
        ofs << headers;
        ofs << endl;
    }
    else
    {
        ofs.open(this->path, ios::app);
    }

    string line;
    this->content >> line;
    while (!this->content.eof())
    {
        ofs << line << endl;
        this->content >> line;
    };

    ofs.close();
}
