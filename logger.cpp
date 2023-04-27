#include "logger.hpp"

logger::logger(string path, string columns[])
{
    this->path = path;
    for (int i; i < columns->length(); i++)
    {
        this->headers[i] = columns[i];
    }
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
        for (int i; i < this->headers->length(); i++)
        {
            ofs << headers[i] << ",";
        }
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
