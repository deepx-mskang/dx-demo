#include "string.h"
#include "bbox.h"

BoundingBox::BoundingBox(unsigned int _label, std::string _labelname, float _score,
                         float data1, float data2, float data3, float data4)
    : label(_label), score(_score)
{
    box[0] = data1;
    box[1] = data2;
    box[2] = data3;
    box[3] = data4;
    labelname = _labelname;
}

void BoundingBox::Show(void)
{
    std::cout << "    BBOX: " << name << ", " << labelname << "("
         << label << ") " << score << ", ("
         << box[0] << ", " << box[1] << ", "
         << box[2] << ", " << box[3] << ")" << std::endl;
}
