"""
Instrument control for the Xilinx ZCU111 RFSoC with primecam readout firmware
"""
import sys
import os
import posixpath
import socket
import shutil
import paramiko
import numpy as np

class RFSOC:
    def __init__(self, out_directory, bid = 1, drid = 1,
                 udp_ip = '192.168.3.40', noiseq = True,
                 local_primecam_path = '~/github/primecam_readout/'):
        """
        Send commands to the rfsoc and save the output data.

        Parameters:
        out_directory (str): directory to transfer the saved data
        bid (int): board identification number
        drid (int): drone identification number
        udp_ip (str): IP address for noise streaming
        noiseq (bool): If False, doesn't set up noise streaming
        """
        # Create tmp and log directories
        directory = os.path.normpath(os.getcwd())
        self.tmp_directory = os.path.join(directory, 'tmp')
        self.log_directory = os.path.join(
            os.path.dirname(os.path.dirname(directory)),
            'logs',
        )
        for d in (self.tmp_directory, self.log_directory):
            os.makedirs(d, exist_ok = True)
        # Import functions from primecam_readout
        local_primecam_path = os.path.normpath(
            os.path.expanduser(local_primecam_path)
        )
        sys.path.insert(1,
            os.path.abspath(os.path.join(local_primecam_path, 'src')))
        sys.path.insert(1,
            os.path.abspath(local_primecam_path))
        from queen import alcoveCommand
        from alcove_commands.tones import genPhis
        from alcove import comNumFromStr
        self.comNumFromStr = comNumFromStr
        self.alcoveCommand = alcoveCommand
        self.genPhis = genPhis
        # Set system variables
        self.bid = bid
        self.drid = drid
        self.out_directory = os.path.normpath(os.path.expanduser(out_directory))
        os.makedirs(self.out_directory, exist_ok = True)
        # Bind socket for noise
        if noiseq:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((udp_ip, 4096))
        self.sample_time = 5 / 2441

    def set_nclo(self, frequency):
        """
        Sets the LO frequency

        Parameters:
        frequency (float): in MHz. 1 MHz resolution
        """
        self.lo_freq = frequency
        com_num = self.comNumFromStr('setNCLO')
        args = str(int(round(frequency, 0)))

        with hidePrints():
            response = self.alcoveCommand(com_num, bid = self.bid, drid = self.drid,
                                     all_boards=False, args=args)

    def write_vna_comb(self):
        """
        Writes a rough vna comb of 1000 tones with a 500 MHz bandwidth centered
        on the LO frequency
        """
        com_num = self.comNumFromStr('writeNewVnaComb')
        args = None

        with hidePrints():
            response = self.alcoveCommand(com_num, bid = self.bid, drid = self.drid,
                                     all_boards=False, args=args)

    def write_targ_comb_from_vna(self, f_filename = False, a_filename = False,
                                 p_filename = False):
        """
        Writes a target comb from the most recent vna sweep

        Parameters:
        filenames (str or False): Path to save the output files. Must end in
        .npy. If False, doesn't transfer file
            f_filename: list of resonators. Convention includes 'f_res_targ'
            a_filename: list of amplitudes. Convention includes 'a_res_targ'
            p_filename: list of phases. Convention includes 'p_res_targ'
        """
        for filename in [f_filename, a_filename, p_filename]:
            if filename and not filename[-4:] == '.npy':
                raise ValueError(f'filename must end in .npy')
        com_num = self.comNumFromStr('writeTargCombFromVnaSweep')
        args = None

        with hidePrints():
            response = self.alcoveCommand(com_num, bid = self.bid, drid = self.drid,
                                     all_boards=False, args=args)
        for s, filename in zip(['f_res_targ', 'a_res_targ', 'p_res_targ'],
                               [f_filename, a_filename, p_filename]):
            if filename:
                self.transfer_file(s, filename)
        if f_filename:
            self.fres = np.load(os.path.join(self.out_directory, f_filename))
        if a_filename:
            self.ares = np.load(os.path.join(self.out_directory, a_filename))
        if p_filename:
            self.pres = np.load(os.path.join(self.out_directory, p_filename))

    def write_targ_comb_from_targ(self, f_filename = False, a_filename = False,
                                  p_filename = False):
        """
        Writes a target comb from the most recent target sweep

        Parameters:
        filenames (str or False): Path to save the output files. Must end in
        .npy. If False, doesn't transfer file
            f_filename: list of resonators. Convention includes 'f_res_targ'
            a_filename: list of amplitudes. Convention includes 'a_res_targ'
            p_filename: list of phases. Convention includes 'p_res_targ'
        """
        for filename in [f_filename, a_filename, p_filename]:
            if filename and not filename[-4:] == '.npy':
                raise ValueError(f'filename must end in .npy')
        com_num = self.comNumFromStr('writeTargCombFromTargSweep')
        args = None

        with hidePrints():
            response = self.alcoveCommand(com_num, bid = self.bid, drid = self.drid,
                                     all_boards=False, args=args)
        for s, filename in zip(['f_res_targ', 'a_res_targ', 'p_res_targ'],
                               [f_filename, a_filename, p_filename]):
            if filename:
                self.transfer_file(s, filename)
        if f_filename:
            self.fres = np.load(os.path.join(self.out_directory, f_filename))
        if a_filename:
            self.ares = np.load(os.path.join(self.out_directory, a_filename))
        if p_filename:
            self.pres = np.load(os.path.join(self.out_directory, p_filename))

    def write_targ_comb_from_custom(self, fres, ares = None, pres = None):
        """
        Writes a target comb from custom lists. Three custom lists must be
        stored in the alcove_commands folder on the board:
           custom_freqs.npy, custom_amps.npy, and custom_phis.npy

        Parameters:
        fres (np.array): Array of custom frequencies in Hz
        ares (np.array or None): Array of custom powers, or None to auto
            generate equal powers at the maximum amplitude
        pres (np.array or None): Array of custom phis, or None to auto generate
            random phis. Checks to ensure that the waveform doesn't saturate
        """
        self.make_custom_tone_lists(fres, ares, pres)
        self.fres = np.load(os.path.join(self.tmp_directory, 'custom_freqs.npy'))
        self.ares = np.load(os.path.join(self.tmp_directory, 'custom_amps.npy'))
        self.pres = np.load(os.path.join(self.tmp_directory, 'custom_phis.npy'))
        com_num = self.comNumFromStr('writeTargCombFromCustomList')
        args = None

        with hidePrints():
            response = self.alcoveCommand(com_num, bid = self.bid, drid = self.drid,
                                     all_boards=False, args=args)

    def vna_sweep(self, filename, npoints = 500, N_accums = 5):
        """
        Executes a vna sweep using the current tone list

        Parameters:
        filename (str or False): Path to save the output file. Must end
            in .npy. Convention includes 's21_vna'. If False, doesn't
            transfer file
        npoints (int): number of points per tone
        N_accums (int): number of accumulations to average per point
        """
        if filename and not filename[-4:] == '.npy':
            raise ValueError(f'filename must end in .npy')
        com_num = self.comNumFromStr('vnaSweep')
        args = f"N_steps={npoints}, N_accums={N_accums}"

        with hidePrints():
            response = self.alcoveCommand(com_num, bid = self.bid, drid = self.drid,
                                     all_boards=False, args=args)
        if filename:
            self.transfer_file('s21_vna', filename)
            separate_iq_data(os.path.join(self.out_directory, filename))

    def target_sweep(self, filename, npoints = 500, bandwidth = 0.2, N_accums = 5):
        """
        Executes a target vna sweep of the current comb

        Parameters:
        filename (str or False): Path to save the output file. Must end in
            .npy. Convention includes 's21_targ'. If False, doesn't transfer
            file
        npoints (int): number of points per tone
        bandwidth (float): Span around the tone for the target sweep in MHz
        N_accums (int): number of accumulations to average per point
        """
        if filename and not filename[-4:] == '.npy':
            raise ValueError(f'filename must end in .npy')
        com_num = self.comNumFromStr('targetSweep')
        args = f"N_steps={npoints},chan_bandwidth={bandwidth},N_accums={N_accums}"

        with hidePrints():
            response = self.alcoveCommand(com_num, bid = self.bid, drid = self.drid,
                                     all_boards=False, args=args)
        if filename:
            self.transfer_file('s21_targ', filename)
            separate_iq_data(os.path.join(self.out_directory, filename))

    def capture_noise(self, seconds):
        """
        Captures noise data. Sample rate is 488.2 Hz

        Parameters:
        seconds (float): capture time in seconds

        Returns:
        i, q (np.array, np.array): 2D arrays, where each item corresponds to the
            i or q data, respectively, of a given resonance.
        """
        packetrate = 512e6/2**20 # packets/sec
        i, q = getNpackets(self.sock, int(packetrate * seconds))
        l = 0
        for d in ['fres','fcal']:
            if hasattr(self, d):
                l += len(getattr(self, d))
        i, q = i[:l], q[:l]
        return i, q

    def capture_save_noise(self, seconds, filename):
        """
        Captures and saves noise data

        Parameters:
        seconds (float): capture time in seconds
        filename (str): name of the file to save. Must end in .npy
        """
        if not filename[-4:] == '.npy':
            raise ValueError(f'filename must end in .npy')
        data = self.capture_noise(seconds)
        np.save(os.path.join(self.out_directory, filename), data)
        tsample_name = filename.replace('.npy', '') + '_tsample.npy'
        np.save(os.path.join(self.out_directory, tsample_name), [self.sample_time])

    def find_vna_res(self, filename):
        """
        Finds resonators from the most recent vna sweep, using the built-in
        algorithm (I would reccomend doing this on your own instead)

        Parameters:
        filename (str or False): Path to save the output file. Must end in
            .npy. Convention includes 'f_res_vna'. If False, doesn't transfer
            file.
        """
        if filename and not filename[-4:] == '.npy':
            raise ValueError(f'filename must end in .npy')
        com_num = self.comNumFromStr('fineVnaResonators')
        args = None

        with hidePrints():
            response = self.alcoveCommand(com_num, bid = self.bid, drid = self.drid,
                                     all_boards=False, args=args)
        file, _ = self.get_recent_file('f_res_vna')
        self.fres = np.load(file)
        self.transfer_file('f_res_vna', filename)

    def find_targ_res(self, f_filename = False, a_filename = False,
                      p_filename = False):
        """
        Finds resonators from the most recent target sweep using the built-in
        algorithm (I would reccomend doing this on your own instead)

        Parameters:
        filenames (str or False): Path to save the output files. Must end in
            .npy. If False, doesn't transfer file
        f_filename: list of resonators. Convention includes 'f_res_targ'
        a_filename: list of amplitudes. Convention includes 'a_res_targ'
        p_filename: list of phases. Convention includes 'p_res_targ'
        """
        for filename in [f_filename, a_filename, p_filename]:
            if filename and not filename[-4:] == '.npy':
                raise ValueError(f'filename must end in .npy')
        com_num = self.comNumFromStr('findTargResonators')
        args = None

        with hidePrints():
            response = self.alcoveCommand(com_num, bid = self.bid, drid = self.drid,
                                     all_boards=False, args=args)
        for s, filename in zip(['f_res_targ', 'a_res_targ', 'p_res_targ'],
                               [f_filename, a_filename, p_filename]):
            if filename:
                self.transfer_file(s, filename)
        if f_filename:
            self.fres = np.load(os.path.join(self.out_directory, f_filename))
        if a_filename:
            self.ares = np.load(os.path.join(self.out_directory, a_filename))
        if p_filename:
            self.pres = np.load(os.path.join(self.out_directory, p_filename))

    def close_socket(self):
        """
        Closes the connection to the socket
        """
        self.sock.close()

    def get_recent_file(self, file_type):
        """
        Gets the most recent file name that contains a given string from the
        tmp directory

        Parameters:
        file_type (str): string that the file must include

        Returns:
        path (str or None): path to the file or None if not found
        filename (str or None): name of the file or None if not found
        """
        files = [os.path.join(self.tmp_directory, f)
             for f in os.listdir(self.tmp_directory)]
        files = [f for f in files if file_type in f]
        files.sort(key=lambda x: os.path.getmtime(x))
        if len(files):
            path = files[-1]
            filename = os.path.basename(path)
            return path, filename
        return None, None

    def transfer_file(self, file_type, filename):
        """
        Finds the most recent file of the given type and transfers
        it to out_directory, saved as filename.

        Parameters:
        file_type (str): string that the file must include
        filename (str): transferred file name. Must end in .npy
        """
        file, filename0 = self.get_recent_file(file_type)
        if file is None:
            raise Exception(f'{file_type} data not found in tmp directory')
        shutil.move(file, os.path.join(self.out_directory, filename))

    def clear_tmp_directory(self):
        """
        Clears all .npy files form the temporary directory
        """
        files = [f for f in os.listdir(self.tmp_directory) if '.npy' in f]
        for file in files:
            os.remove(os.path.join(self.tmp_directory, file))

    def clear_tmp_directory_full(self):
        """
        Clears all files from the temporary directory
        """
        files = os.listdir(self.tmp_directory)
        for file in files:
            os.remove(os.path.join(self.tmp_directory, file))

    def transfer_custom_tone_lists(self):
        """
        Transfers the custom tone lists from tmp_directory to the board
        """
        # Set up SSH client
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # Connect to the xilinx board
        hostname = '192.168.2.98'
        port = 22
        username = 'xilinx'
        password = 'xilinx'
        ssh.connect(hostname, port, username, password)
        # Transfer files to the board
        scp = ssh.open_sftp()
        files = ['custom_freqs.npy', 'custom_amps.npy', 'custom_phis.npy']
        remote_directory = '/home/xilinx/primecam_readout/src/alcove_commands'
        for file in files:
            local_path = os.path.join(self.tmp_directory, file)
            remote_path = posixpath.join(remote_directory, file)
            scp.put(local_path, remote_path, confirm = False)
        # Close connection
        scp.close()
        ssh.close()

    def make_custom_tone_lists(self, fres, ares = None, pres = None):
        """
        Creates custom amplitude and phi lists from the give tone list, and
        saves all three files to the board

        Parameters:
        fres (np.array): Array of custom frequencies in Hz
        ares (np.array or None): Array of custom powers, or None to auto
            generate equal powers at the maximum amplitude
        pres (np.array or None): Array of custom phis, or None to auto generate
            random phis. Checks to ensure that the waveform doesn't saturate
        """
        amp_max = (2 ** 15 - 1)
        fres = np.array(fres)
        if ares is None:
            N = len(fres)
            ares = np.ones(N) * amp_max / np.sqrt(N) * 0.25
        if pres is None:
            pres = self.genPhis(fres * 1e-6, ares)
        for file, res in zip(['custom_freqs.npy', 'custom_amps.npy',
                              'custom_phis.npy'],
                             [fres, ares, pres]):
            np.save(os.path.join(self.tmp_directory, file), res)
        self.transfer_custom_tone_lists()

def separate_iq_data(path):
    """
    Given a path to IQ data saved by the RFSoC as complex f, z,
    split the data into float f, i, q and save it

    Parameters:
    path (str): path to the saved data
    """
    f, z = np.load(path)
    f = np.real(f)
    i, q = np.real(z), np.imag(z)
    np.save(path, [f, i, q])

class hidePrints:
    """
    Disables print statements
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def capturePacket(sock):
    """
    Captures a packet from the noise streaming ethernet port

    Parameters:
    sock (socket.socket): socket for noise streaming
    byteshift (int): byteshift for rolling the data

    Returns:
    packet (np.array): captured data
    """
    byteshift = -1
    data = sock.recv(9000) # buffer size is 1024 bytes
    data = bytearray(data)
    data = np.roll(data,byteshift)
    return np.frombuffer(data, dtype="<i").astype("float")

def getNpackets(sock, N):
    """
    Captures N packets and converts to I and Q

    Parameters:
    sock (socket.socket): socket for noise streaming
    N (int): number of packets to capture

    Returns:
    I (np.array): each element is an array of I values for the respective tone
    Q (np.array): each element is an array of q values for the respective tone
    """
    p = np.array([capturePacket(sock) for p in range(N)])
    I, Q = p[:, 16::2].T, p[:, 17::2].T
    return I, Q
