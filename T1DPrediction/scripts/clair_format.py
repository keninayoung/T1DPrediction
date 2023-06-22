#!/usr/bin/env python3

import os
import sys
import argparse

#------------------------------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------------------------------
def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('--input', help="Input data")
    parser.add_argument('--type', help="'static' or 'temporal'") # either static or temporal
    args = parser.parse_args(argv)

    """
    subjuid	hla_6grps	hla_6grps_ref_DR44	rs3842727_C	rs6897932_A	veg_solid_food_day
    """
    if args.type == "static":
        out1 = open("static_vars.txt", "w")

        with open(args.input)as f:
            first_line = f.readline()
            first_line = first_line.strip().split(",")
            static_vars = first_line[1:]
            header = ["id"] + static_vars

            # print header
            print(",".join(header), file=out1)

            for line in f:
                fields = line.strip().split(",")
                print(",".join(fields), file=out1)


    # temporal
    #  subjuid	due_num	IL8	VEGFA	MCP3	CDCP1
    else:
        out2 = open("temporal_vars.txt", "w")
        header2 = ["id", "time", "variable", "value"]

        print(",".join(header2), file=out2)

        with open(args.input)as f:
            first_line = f.readline()
            first_line = first_line.strip().split(",")
            temp_vars = first_line[2:]

            for line in f:
                fields = line.strip().split(",")
                id, time = fields[0], fields[1]
                vars = fields[2:]

                for index, var in enumerate(temp_vars):
                    # print temporal variables
                    print(','.join([id, time, var, vars[index]]), file=out2)


if __name__ == '__main__':
    main(sys.argv[1:])
