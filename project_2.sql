PGDMP                          x        	   project_2    12.2    12.2     #           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            $           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            %           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            &           1262    34520 	   project_2    DATABASE     �   CREATE DATABASE project_2 WITH TEMPLATE = template0 ENCODING = 'UTF8' LC_COLLATE = 'English_United States.1252' LC_CTYPE = 'English_United States.1252';
    DROP DATABASE project_2;
                postgres    false            �            1259    34542 
   GDP_states    TABLE     �   CREATE TABLE public."GDP_states" (
    state text,
    year bigint,
    "GDP" text,
    latitude numeric,
    longitude numeric,
    pop_1990 bigint,
    pop_2000 bigint,
    pop_2010 bigint,
    pop_2018 bigint
);
     DROP TABLE public."GDP_states";
       public         heap    postgres    false            �            1259    34530 
   final_data    TABLE     <  CREATE TABLE public.final_data (
    state character varying(150),
    year integer,
    "GDP" money,
    latitude numeric(10,2),
    longitude numeric(10,2),
    pop_1990 integer,
    pop_2000 integer,
    pop_2010 integer,
    pop_2018 integer,
    unemp_rate numeric(10,2),
    mortgage_interest numeric(10,2)
);
    DROP TABLE public.final_data;
       public         heap    postgres    false            �            1259    34527 
   gdp_states    TABLE     �   CREATE TABLE public.gdp_states (
    state character varying(150),
    year integer,
    gdp numeric(10,2),
    latitude numeric(10,2),
    longitude numeric(10,2),
    pop_1990 integer,
    pop_2000 integer,
    pop_2010 integer,
    pop_2018 integer
);
    DROP TABLE public.gdp_states;
       public         heap    postgres    false            �            1259    34533    lat_lon    TABLE     {   CREATE TABLE public.lat_lon (
    state character varying(150),
    latitude numeric(10,2),
    longitude numeric(10,2)
);
    DROP TABLE public.lat_lon;
       public         heap    postgres    false            �            1259    34524    mortgage_interest    TABLE     a   CREATE TABLE public.mortgage_interest (
    year numeric,
    mortgage_interest numeric(10,2)
);
 %   DROP TABLE public.mortgage_interest;
       public         heap    postgres    false            �            1259    34539    state_pop_30m    TABLE     �   CREATE TABLE public.state_pop_30m (
    state character varying(150),
    pop_1990_m integer,
    pop_2000_m integer,
    pop_2010_m integer,
    pop_2018_m integer
);
 !   DROP TABLE public.state_pop_30m;
       public         heap    postgres    false            �            1259    34536    state_pop_in_m    TABLE     �   CREATE TABLE public.state_pop_in_m (
    state character varying(150),
    pop_1990_m integer,
    pop_2000_m integer,
    pop_2010_m integer,
    pop_2018_m integer
);
 "   DROP TABLE public.state_pop_in_m;
       public         heap    postgres    false            �            1259    34521    unemployment    TABLE     S   CREATE TABLE public.unemployment (
    year money,
    unemp_rate numeric(10,2)
);
     DROP TABLE public.unemployment;
       public         heap    postgres    false                       0    34542 
   GDP_states 
   TABLE DATA           w   COPY public."GDP_states" (state, year, "GDP", latitude, longitude, pop_1990, pop_2000, pop_2010, pop_2018) FROM stdin;
    public          postgres    false    209   �                 0    34530 
   final_data 
   TABLE DATA           �   COPY public.final_data (state, year, "GDP", latitude, longitude, pop_1990, pop_2000, pop_2010, pop_2018, unemp_rate, mortgage_interest) FROM stdin;
    public          postgres    false    205   >+                 0    34527 
   gdp_states 
   TABLE DATA           s   COPY public.gdp_states (state, year, gdp, latitude, longitude, pop_1990, pop_2000, pop_2010, pop_2018) FROM stdin;
    public          postgres    false    204   �A                 0    34533    lat_lon 
   TABLE DATA           =   COPY public.lat_lon (state, latitude, longitude) FROM stdin;
    public          postgres    false    206   �A                 0    34524    mortgage_interest 
   TABLE DATA           D   COPY public.mortgage_interest (year, mortgage_interest) FROM stdin;
    public          postgres    false    203   �D                 0    34539    state_pop_30m 
   TABLE DATA           ^   COPY public.state_pop_30m (state, pop_1990_m, pop_2000_m, pop_2010_m, pop_2018_m) FROM stdin;
    public          postgres    false    208   nE                 0    34536    state_pop_in_m 
   TABLE DATA           _   COPY public.state_pop_in_m (state, pop_1990_m, pop_2000_m, pop_2010_m, pop_2018_m) FROM stdin;
    public          postgres    false    207   �E                 0    34521    unemployment 
   TABLE DATA           8   COPY public.unemployment (year, unemp_rate) FROM stdin;
    public          postgres    false    202   EH              x��ێGr��KO�H6��q)��޵���#��E�rM����O�?��{fȦ{�2Hr�(�=_WUf3���n�����}(��w���)�M�D���6,!�)��S�)�������k���o�X�%W�����)��M�f�)��8�M�f��wBuW��G_��6�'
�Ğ�߆q�9�J�>��"IJ�L�8���CS<��O���S	���	}�:����{9�9�LZ�y������ǅ]���5K����&���1%���!�%��V0񝙞�hJ!8/a����]��l��V���B�Y7M��.T��z����a{��!���e�!a������%�V���S�Y���R����� 8��KcВK�m�nZ��j{p�4�BJX�������������\)�1Ƕ0�@�ܺ���
%5��=�yJK.��$��>���;*����a\�B�-D�"&8�W��Çaa2.$4Is�������~^��)%� pQ ��l��6���f�+!�``I
H����u�F�p�cp�-Rw�k�:K�D�8,�xa���������ݧ��|���"�a�qz���0+$)bY�1g_HS��9d}N�/?�p(9�8������X�8��>�@dQ� ��o��7w��7󊰦d�l@���[�|��'"
��TSA���$�A��&W����zE�Ħ>�\�a1�� �EH�_�v���������$��"�5���O!��7[��<	� M-��d��lb.�T��2�%~è��);��Bg�>#�'?�	 �g$^�۲_K�x���e��d���>�����*a��Q��08�X�?jE �ƍ�;I��&|���A@�D=5�@���I\L4
)����� ��{��p�d���E��A�HA�iC�C�'B�P��P5ǆ)����_�1��
w\bY��>�@����C�ՠDJ+O�i�J�\'���>q�[�#:T���E�s���j�ɛ,5%���k��a��,�B��[`��g	�X4aל6��ŰZ�c�zY�R�,��0G�y^�������b�iΈe�p&(���Y$׊�%x�j&7ÒD!&��`�	Ղ6R��J����!bR�։B$e�KԽՉ�^FzB��nw��-U8��>���%�o+�[������D���5a��� �տ b�{��[!� "��c��!�:U*-%u�-Gr�������n{�R���"�E�l*��Zg1���ddp�+�I@���`J*��0��W7�P�0ͺ�ӊ��a�5���K&�Cu3�Y��xLe�"��U�6 d��� whd&,�����͕Lk�����HB��ZEP7	�j^�����>t_|�d��A`��7Kl-�9��Oc�P���Hkcf.�j5[%�S�o�3,�9��幪����gN%��~�!ᙷ*��FP	�U bh3�����O�>��<s	N�@�PNdL�k�uVE�1��+��"�r�;ʋ�g��	�všU�0<_���Y'L��d�Rx�m��������#
Q�mׇM ���z������b@ ����R����|��9"+���8��O�G��a��-��㤵�j�ո?mw�K��%����9e�o�M�a�(�4�B�f�&<�7�f<5?�P}����a�#!<E�5��%U���|/	���&=�t����8�y��9�{�Ǻ�����YN%
���"qb����8�4$9wYeIr�h0cD6���s�N9DWj^��>�`_#���k;��n?n���-�z��G�������Dᄐ�&Be]t}=�-|e�/�0�����e@�F2�B�qWY���rx�P\)Ǥhr��v�a{?�L���s�VA�F��:{����2I��:�$�i�Zs�V�H�P^�t: ;v�4��ǝ�g��4,ETw���iX��
�m�u�������X�E����}$�[�+߆p��}�B�x��I�/<P�%N+2A���Y���,�	�qZ�Ux���\7uY���˕��A��ᇥU���>-�@(�I |?�>	i����7�7=dE����q�?"u�s$�1���B6�
�6\My�4��<��8�Z�����8T�BP�R(������u"�v@z��T��{��o"�&��4j#:(ه�X�N'�G7���w�a)��h0��$w������ZBω`�jf��5���Y	�[B>�"f�0��
Or�BTyNB�(.�'���pK��HU��ś?�"!4D������滇��=ո�9}@	KB>,u���:�*�|?=i9Je��0x��}�ؠ�m"�^ջ`�D	�r���8�D�@�
�_��͐����tzR@lK����ǃe�a=���:-�aw>eĘg0mi��,��Ӷ;Bh�Ņu��y$�e9W�i����se?���������aE[	��R�8g�������1mu�R�(i��7%��R�$����B�vp��+�"�	g�4�e�աzh�d-��n7K{O���r*��-:+���h�&�*Yu��ʚ��	�C��?㡤�ѡ���V)���P�FDS���^�L���.�D�x�����r�)�h�o��5���<R�\ss}�xm�� s��"�`A���&s>�G�x�}�M����n^;�t����n���c��_���K�@���m����N���\�����̄�G�r'�!fmwg$򂉿wm����iF0a�p�!��������*�h�u�h��d9Y]��?<޾���a�;���'Gȳt'��L�H�շKH����G2\��GF!#%NZ�%��j�_@�U�s�=
����õQP�R�P�|L:�4X��<J�����KR�� !�)�J@�� �����*�������Qc"ғǎ�����x�~7�5���4 ��ъ�0ѣ͆�
�?��4���4���E	�<
{��:&HFC+XP�$]�P�<mU*�O4jq��ك���������!�sA7�ڰ��Zj����v|�.T\_�O�
��rl��K�S
���\|�GL��Uu�_g���� ���v��Am��a MOUc{aM�/U鸶����g���/�C8��:�W��K��Ԋ:���ȲQ��Q���y��&|旑�.�H(��\Ym��\�t���_Z��r�U����?�t�����"�C���+�.#�si��U��!����}��#��o<�/��}x�a)�FQyb�$�ܐEm���������&W�������͛?�NcԸ��/�V�����[*�Z��f�6B�x�GN�<ҸF���Q7���ǟ��4�!Z^���!�`Q�������O_$���$���Z�\�[}�>Ы%VJm�Xk�2Ed��
��g��͢��� ��:5�����L�mz�2����2��k'�y���z���e*�s���,.	�<���yt���oϫ|�)W8�6��c}�S]Z�`z2��>/A��|7aʪ+��=W<���d#�ش�Z��1�6�I�&��"%�i҅�W,�����*O�K���"��C@�8�MH�,%k'N�dsn���L6$����4���v^�� "~k�ڷ���J��j	�62��E��*w����P�8Z�����Ҁa�j�j�+��I���2����������aI�[�����b����6��h����bFZ��O(=�v؞zQ:N�.��U�F��QҪ��]����y�~���P�GY&���/��]��I_(��E�}/ �vA⩢p��9*,��2�������e�Vs����P��ǣ�e�rWrvv��"Wd�a��з�D���\uT��5}��I�T�o!;��0�򂦃Q��������Qj~��@/�k�Te_PJn�������nw���5=�������aP�d,��I�N�\`l\����p�QKKOf P{����sP���4��5 �#��܏[R�xsx|��>ط��9A-Hs���R�v�Sf�	��sꡖ���?��� z  ��nǰ� ���(.����M�������'�*{S�w�w������u-m���i�
�:-z���]Ƣ�� C.Uy�K�F	T�� D?N�Eh�!��i� %_V��i�`��
�z�O��au�Bp�̥T��W��l05�(,mQ�.=�Ҩ���Rs���@q�L�wpX��]�Vs��U��ޗJ�8�����8߾^R�/��Hm�U�K"s�����VA/��]5�9ӟ3��/t�Pn̿��Э�^ң[G��K��zI�������}�L�u��_S������(�״�:��D��q��Xu�ܳ2w�g��p�
߭��)���
�S^��(�u�/*�u�/*�u�/*佊~I)��%?I�=e3#��~�Y-����q��������0��_��W��ḇS�o��@�0�9ža�s~���k��|�8m��5��G^���yQ���H��3�?��3�?��3�?��3�?��;��G�,8�D�S�D���D��\��B(��{�`�*�E��Uԋ���������YP�T�ӓB^>�Ӻ���J�ß� �?+J؋7�BS)4���*�'S+W؏=�[؏=+`؏=�d�k���4I��O��Z���)Fu��ß�=�?+�8V�ob�&�hb�&��[K|:�+���{Q<q%����J�e1ū�&��[U<Y��]�K"�]�Kb�]�K��W�M|�*��,k��؍=/�؍=��؍=/иkJ��.��cs|�b��E��5Ћ
�k��_5EGSt4EǟY����+OR����Q�������/9+9�KΪB�����u�D>����\��$�{I>r%����5��[N�d}u��� �W&���49�������kO>��iP^���Ey5�&����ڔ�F�F�/P��zC�*WQ/jV��^Ԯ\E��a�j�iY���iY���oY��t�4J��z�u��d.�^л|%Є/M�҄/M�҄/M�҄/M�҄/M�҄/M�҄/��8�4�K�4�K�4�K�4�K�4�K�4�K�4�K�4�K�4�K�4�K�4�K�4�K�4�K�4�K�4�K�4��+�&|i&|i&|i&|i&|i&|i&|i&|i&|i&|i&|i&|i&|ik�&|i&|i&|i&|i&|i&|i&|i�A��	_��	_��	_��	_��	_���z�	_��	_��	_���uH��4�K�4�K�4�K�4�K�4���H�4�K�4�K�4�K�4�K�4���&|i&|i&|i&|i&|i&|i&|i&|ى5�K�4�˫�&|M�҄/M�҄/��Є/M�҄/M�҄/M�҄/M�҄/M�҄/M�҄/M�҄/M������o�����d��            x��K�Gr��W��/��|?���#�c~	��"{��}����_�QU�2�-��w��aVfdD�WY���?���O޺t�;o���OŞ���;��y*�m�������K�.�S����NeJ锒�l���ϷO�`�&G7e{�n
����E�?km��s�T����J(���������b�<�� ����s
���P*���R���N����v�-��US���!6O#�!�����|'�\���=�O��ϟ/�>=^�5)S�g�<9OCmS�Щ�E_��9��O1y�GK�ظӾ~�>�?\�ᦄ�B�M6Cצ)8�W��S�ؚO�b�?f���|||x�|y��yp��b�X�y���'%C��J�6�3��0���c��?=|>�z~zX$�5-7Zl4`J��;���ԐR;�VZ���i�F�����������o0_��t�]��Xx��J�����"�ln�Ux�u%c�i��cF/V���*�J�2@�����U�Q�԰���bu*�~��]��������3FS�P��a�2dl�X��R��ɱ�,�5�\�e��3��rY����d韟�7���g��9[���"��.d�Gd6�qR��p��FR�{p
X�8!��q�#�����tR.�~,	������������Ҽi��<�)$<4�Cs�J�hF����0~��Ŝ��.�����Ԧ�8)yDv���BW� 6}��f��搖F����)��%��{�q-L�a�KƠ�A=�6z�#�8��m�wCI� sM��k��eJNHD%!	y,qC����ø�I\�;�����?��J6g��8�L�b2�!�-���5���?b��ۃ�?]�^n�4���&G��M��{W1,h4�5�Ȥ�sjm7�9_����I>Q�@�`���fǴ9����t8��@�	����=�����a]�\��ܥ{n+yBhP�)���[���G�h�<��#����?~�=<?��Eț�*m�8���h��!r���[���h�(U�zڣ���������E2�hP	8ұL�(!����2OH��_j�F���-5�v��·��=�}3�b��g?i��D���eh
6P<!,!���F���5m ������JDsJ�υ�e#�ܢ0q}B���ׯO�{���ȹ�N(�T2=U_��� S�Tr�7�\=J��(p����y�͘5��yj\,�D{�5�'�z�3�d�ȩ2���?����S�v`���8b��d�(�����\�4N�$��������^$\��	ܺ���a�u��\H���J��#D��N��7>����UId��q�'��̔ԱĔ
%u�c��&6���|x�=��%�`��(S}Dpb,�#�J8�j���(¶&o_�������Z.�\�-�I�ʅEƃ'$3���a��7Y:�H�P/^����O��I5��ࣜ��I9	#m��/H>*8&�N�)G���O�����tE=�z�`2��R�2e���zڜ��(t(��������-n��y`�?�?m�4%c��X�t�&5c�cv�Q<�ǌ��S��1(���r/��r����u� FD(���N��
�yB�7��>��߻xW�AB�.=WႄjP�hP!&�1h֮R߉�K$?<=�|]���L�O���,�,?!,�<|�������:�jl:,�]��/����>�����xF�xY����ȡ#U��bRm�\M�|���q��� ����|�jI���q��x_9jn�#�]�����X��.?����ˠ��h�r�&��o���E�Dn��@F@�٣�@.�Rb����S���_�1ڱ�E�"5�j&� ���C��� +���@����p/%� �9C�D�-6)��cTR
"j�0�zel2��N���⠗Eׄ��r�����NT�"`8h���O���!8�uu�9�����=�6�M|����W�4�Ѹ�e�\��r�O_����""/�z�����d}n�'�X1� ��E��5��ѡ��HgO�C�����'q��<��ɠTJRYɡ�.�4�$�T0h!���xF�����u���R�O�%��68a�-����~�
�����p{~�-t��x�*�-��者�#'Wj��K?���ȏ�������x�N��2:#��w@�IϏ�9)PRJ�(W��S}s���]�`VE�97q�/|z-<xH�������Q��FoV\���X1�U�+2m	�)d�� Ed�^�8��6\������K"���+2�}-6�W��#��y48ࡥ+�b\1hw�"�=--|��Qs�4��Ҏ����W�q$�3�ظf�����
^��(������;\5,�3�W�^��PM��i�IqE/��
(��hi�W�w\1�B�W\�+�W�m6(�k�,���+(�@nYn	�\A�S(0�W�r��֨��e��A��$�p܊3����^r���}1��M�W��w\A��a���=�+F�W�B�=�T-}W��;\A#u�D[�?�+�Wp]sk��Q����`���M�+F�WP(ٙe!�ث��"�oR�b\1Hn���3q/,��\�+�]6�<\1J����8V.-%G�+F���0����%�b/��
���v�W�E7\Ae�W�@"\�]qE�CUɔxIq� �W,۞��N�+^*oG�̯�*j���^q�T�q�Dď/��d�+��c�)���WІw��d�^A�+z�=����ئyNrR\�K�qEf����)���K\�S�M�s������CL�jнrL���+��`��w	"\1j~�Zz�f���H�+z�WpN-҃W�z��"^�T�A� \�q���W��#�X:��v
pŠ��s?O$ 
q� ��
�΀��3�a\1hn���)sM:�+vpa��6#��9.�b$��"$#�z\Q�d�j9�KpE'��Bg�bbK������pE��o0Ȳ\�d�b�qͣ�T+�s���N��4(��R<�qE�9�
p�8hZ�pd�bS�qEa�WtWtj��+h�BD�4�:�+6�W@�U$�̈��bS�p~��C��vE/��B�@Ѓ��Jq�&x��>�m~�+��^�+h�h?*���*\��n��C�
�,���+(z*rNEA�+6�wC�t�i��M�+:�W���[q�B\�I��4q4��u!��4︢P��Mh�B\��m���E�QɳW��;\A{1b6k߮t;\A�6��s���d�+�8�n~b��5\Aщ���O|�b���
�����ф����pEᗋ�KܭKp�&��2��@��w	��%W\A��"i�e��pE����2�A�Z�H�+F�WPp����b�p�I��\�6I�+F�WФ�����~�+hB58=.��E�b��-I���� ���+(�FZ��7帢��pW�ʯ��Wt�w\A�ug4��^q�+��;Km�r\�I�q-R5K^$��T_�
��`���ĸb/<�T�Y���pŦ8�
����r�H�+z�o[�C����Wl�+�ྩ��G��^��|@D�Ï-��X ������,���+8{D���SqE�������/W�%����qO�c6Wĸ���pE���/�|Wpa��6�ö�ף"\��Y��*���)B/�p� ��J��"��%Wp�(��J��ڜ�E���qͣA��'�C]�+�W��Pl]KW�������k^i�a\ѫ������8���~WВ����}�Kw��Ԍ��\ȯ�n,(��7h2/ ��,*��;b`�ށ�P1��XI�Š�H����5�������٠&,_	��,h�+�����^�ݐC���"�q`1v��6���V^^��%`A�	�A-K�qX��w`Q��65�%~��QnPDggR��w?ǁ�(�u��Z%�n,H�ӭF�"`1Jv�����@��Ũ9 �N�u
�J��^uu�$�������b�܀E�W�tp�"`������o!�rmA,F�X�G�V�I�Ũ��2���f	���n��ί���+7>,������W^� _  
��^u%u>!�s~��A���}O�F��!"/���%�"9ˈE��Z)4^8��� �.����A�N,��K��IP@,z�=����)k}�!#���X��ZԒ�~�|�X��/�O}��߃�c
[��	#��@,��D�G�QJ�Ũ�m�qЁ!��{��^u%u��B��^�#�������(ǈ� �1 Z��ڭ�;N,zɑXp�H(�\=e�bP�����ae�ǉ� ��Ό��C�ab1hnĂ�=~�}��+�h��#ź47ǉE�l�+�#�?���=���bۈE�O.�%W��(��Fo�2��^r$4PڇhD��=��b��)7�J)��+F́X���-�h�߃��=��$Jq��8L,��!�d��s�_%�tG,�V�8��X���h�mj6a�2YF,zɍX@��@�H�E/x'm~�眜Xz� I2���U_�����G*v�� 	�bA+N����u��X�r�Bp;�uMB,��XІ�1�B�EH/9�6�E�AJ,z�;��/�`���qb1�m�����3�+��X���sJ�'�nG,H2:HZ~z	�%;bAq�Ygo"�5b���2WM	�ث�Ă�F��lNH,ɍX4���W�W,z��X�CP
y�1x�X��+��=^LvQ~�bT�J�� <9K��^v#�3c4x�&{эX��h��r����J,��┈*4wRb1� Ud�1�8�x���.m@�UĢW\��T#[��*b1HvĂR���1�$�Ă_�Y�d�^qO,(��ʹ˙UF,z�=����r�"bѫ�$4h?��bb�Rx�)4�6��X��Xj	� )�5�m=��5����%#��J,�|(�5O	�E��zn����e�1b1�u�ϯ���X��#��n�a/��D�bP�����8	�� ��΂�\�T:N,͍X�	'͕P$;����j����j����j����j����j����j����j�yߩj����j����j����j����j����j����j����j����j����j����j����j����j����j����j����j����j����j����j����j����j����j����j����j����n��nJ��n��n��n��n��n��}����j����j����j����j����j����j����j����j����j����j����j����j����j������pS7�pS7�pS7�pS7�pS7�pS7�;U7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�ps�~5�\cJ7�pS7�pS7�pS7�pS7�pS�+�pS7WA5�T��UW7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�pS7�ps]q5�T�M5�T�M5�T�M5�T�M5�T�M5��b����w�}��FM�            x������ � �           x��T�n�0}��b���_�V�RJ�VH����km֮�,e�zf�V���R�H��̙s��|���;)����j0�����}�9Hӝ�A����焙��(�$F�>_:��a�x�]�1��S����Jw�-��c��6w�A�1� 9�S
�9n�s�8�;3�f�a��~
�/��`��2�y��y������@�,8�g�����[�	QY���ې��X'�%h��a�;��w4���g���N)��+P���1�K�z�^�~���6zbŁ*B�u~���#�N����$r�eɮB������4�T�c�1��$N�r��]��!҈R#C�i�i�ʥ�`{���7�c	�\����c_v7���S�$HY
��}�3"juD�N,����H�a��*!��i���±�f�4�*wj`��TW	E�4�a0��o�`�S**�1��z�OeQnD���r�>L%�*�$2"5~~�M���$U����[��4>Q���y�w�?e��D
�_�.���`��nv1׶Ȍ�;y�q;��`��N��)<�D
Q�T߰O!�rzz	XdpuAq!>��6�֥
H+o�i9������`�e����+LU�@�J��@*��Y6W�6�����Ɛ3������Gl5������GO����������\�{��ǙF5h4*�c��̫�wp7�+��X69�X�fY$��O���H$S���<�=͐�!5Cj���R3�fH͐�!5Cj���R3�fH͐�!�א�c��8��         |   x�����0E�u\��4�
����a��#Y��#�㾎����t(�"�*�ܒέ9�ض�ȷ̅����YF�όIj��ќ���C:	kGᳳ�u�����pt�^ �^AY�����p�L         '   x���/O�46 \4�ofq1dT�9������ L�E-         �  x��VM��0='�"��Z�n)G��]��~,��ːx�(��lgY���8��{�4V�<ϼ��I	k� zs����k�I	��hp|�c�Ŀ�����юZ\\������m|%>��#��z�u�6pQ
	�h��b���TcZk�l'񔕰ɂ����S����DRWk�쓉3���l���bh������">v}��E>�����2ȅ�vص�,�T��h`!��m�f<C�]k)0;�+{[���X�ȸ��b���-�WQ�:�"���+>�/�g�v����y��֪/'mn��Y�
�.�0Y	�7���9*���[&���Zb߅NN�M>C�Z��<��\��2���`�������l�_�Tl�����=@��SqE��䷐E4hq�.�XR��=Ha�	ڒ�-��0���m���Y{�}jc^�i��o=�U��m¾1�վ|�fD{�]��s��d�m��Q�'�o��u�W��'kG1���Q�|5Ch`K6l�i7-y~i�=���}b�2]�	Ͱ�g���=u��F�r�|p�xŔNZ�!��P��+����ڋʄ�DDi��1Hct-i��1Hc�� ���E�4���1Hc��x��� �A�̒��G�4����8�Ic�_�ϻ8���.)�         �   x��л�0��:�#e@P��Z"x�9�����*��0����a����K��Y���2l0�ժ٬��Y-��Y��{��W����u��D�|��p`q���K&V�P��웫"��i(��zK)} �\ud     