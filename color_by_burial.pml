load "/Users/famnit/Desktop/pythonProject/pdb1crn.ent", pdb1crn
hide everything, pdb1crn
show cartoon, pdb1crn
color grey70, pdb1crn
select interior_A, (pdb1crn and chain A and resi 3+4+9+10+12+13+16+17+26+27+30+32+44)
color blue, interior_A
select exterior_A, (pdb1crn and chain A and resi 1+2+7+8+15+18+19+20+21+22+23+24+28+29+34+35+36+37+38+39+41+42+43)
color red, exterior_A
select intermediate_A, (pdb1crn and chain A and resi 5+6+11+14+25+31+33+40+45+46)
color yellow, intermediate_A
set cartoon_transparency, 0.2
set ray_opaque_background, off
bg_color white
